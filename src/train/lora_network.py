"""手写 LoRA 卷积层（训练视角）。**不依赖 peft**。

数学形式
--------
对一个卷积核 ``W``（形状 ``[out, in, kh, kw]``），LoRA 学习一个低秩增量：

    y = Conv_W(x) + (alpha / rank) * Conv_B(Conv_A(x))

* ``A``：1x1 卷积，``[rank, in, 1, 1]``
* ``B``：与原卷积同核的卷积，``[out, rank, kh, kw]``
* 复合结果等价于对 ``W`` 的低秩增量 ``ΔW[o,i,kh,kw] = Σ_r B[o,r,kh,kw] · A[r,i]``

因此：
* **初始化时 B=0**，注入瞬间模型输出与原始模型逐位一致；
* 训练时只更新 ``A``/``B``，基础权重全程冻结（``requires_grad=False``）；
* 导出时只需保存 ``A``/``B``（几百 KB ~ 几 MB），永远不动基础模型文件；
* :func:`merge_lora_to_base` 可把增量折叠进基础权重，用于「导出为普通权重」的场景。

具体层实现复用 :mod:`src.lora_runtime`（推理/训练共用的唯一实现，避免两份代码漂移）。
"""

from __future__ import annotations

from src import lora_runtime
from src.lora_runtime import (  # noqa: F401  # 训练侧直接从这里取用
    DEFAULT_EXCLUDE_KEYWORDS,
    DEFAULT_TARGET_KEYWORDS,
    LoRAConv2d,
    count_lora_parameters,
    freeze_base_parameters,
    inject_lora,
    remove_lora,
)

try:  # torch 为可选前置
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

__all__ = [
    "LoRAConv2d",
    "inject_lora",
    "remove_lora",
    "freeze_base_parameters",
    "count_lora_parameters",
    "mark_only_lora_as_trainable",
    "lora_parameters",
    "parameter_groups",
    "merge_lora_to_base",
    "unmerge_lora_from_base",
    "set_lora_scaling",
    "lora_layers",
    "lora_summary",
    "DEFAULT_TARGET_KEYWORDS",
    "DEFAULT_EXCLUDE_KEYWORDS",
]


def _iter_lora_layers(model):
    for name, module in model.named_modules():
        if isinstance(module, LoRAConv2d):
            yield name, module


def lora_layers(model) -> list[str]:
    """返回已注入 LoRA 的模块名列表。"""
    return [name for name, _ in _iter_lora_layers(model)]


def lora_parameters(model) -> list:
    """返回全部 LoRA 参数（用于优化器）。"""
    if not HAS_TORCH:
        raise RuntimeError("训练需要安装 PyTorch。")
    return [module.lora_A.weight for _, module in _iter_lora_layers(model)] + [
        module.lora_B.weight for _, module in _iter_lora_layers(model)
    ]


def mark_only_lora_as_trainable(model, *, verbosity: bool = True) -> tuple[int, int]:
    """冻结基础权重，只让 LoRA 参数可训练。

    Returns:
        ``(lora_param_count, total_param_count)``
    """
    if not HAS_TORCH:
        raise RuntimeError("训练需要安装 PyTorch。")
    for name, parameter in model.named_parameters():
        is_lora = ".lora_A." in name or ".lora_B." in name
        parameter.requires_grad_(is_lora)
    lora_count, total_count = count_lora_parameters(model)
    if lora_count == 0:
        raise RuntimeError("模型中没有 LoRA 参数，请先调用 inject_lora()。")
    if verbosity:
        ratio = 100.0 * lora_count / max(1, total_count)
        print(
            f"[LoRA] 可训练参数 {lora_count:,} / 总计 {total_count:,}（{ratio:.3f}%）"
        )
    return lora_count, total_count


def parameter_groups(model, *, weight_decay: float = 0.0) -> list[dict]:
    """构造 AdamW 参数组：LoRA 权重做衰减，其余（此处只会有 LoRA）不衰减。

    训练时通常只对 ``A`` 施加轻微 weight_decay，``B`` 保持 0 衰减，
    这样初始的零初始化不会被过早破坏。
    """
    if not HAS_TORCH:
        raise RuntimeError("训练需要安装 PyTorch。")
    decay, no_decay = [], []
    for _name, module in _iter_lora_layers(model):
        decay.append(module.lora_A.weight)
        no_decay.append(module.lora_B.weight)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay),
                       "name": "lora_A"})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0, "name": "lora_B"})
    if not groups:
        raise RuntimeError("模型中没有 LoRA 参数，请先调用 inject_lora()。")
    return groups


def set_lora_scaling(model, factor: float) -> None:
    """整体缩放 LoRA 增量（factor=0 等价于临时关闭 LoRA，用于效果对比）。"""
    if not HAS_TORCH:
        raise RuntimeError("训练需要安装 PyTorch。")
    for _, module in _iter_lora_layers(model):
        module.scaling = float(factor) * (module.alpha / max(1, module.rank))


def _delta_weight(module):
    """计算某个 LoRA 层等价的基础卷积增量 ΔW。"""
    with torch.no_grad():
        a_kernel = module.lora_A.weight[:, :, 0, 0]                # [rank, in]
        b_kernel = module.lora_B.weight                            # [out, rank, kh, kw]
        delta = torch.einsum("ri,orkl->oikl", a_kernel, b_kernel)   # [out, in, kh, kw]
        return delta * module.scaling


def merge_lora_to_base(model, *, strict_shape: bool = True) -> int:
    """把 LoRA 增量折叠进基础卷积权重（导出普通权重时使用）。

    合并后模型不再依赖 LoRA 分支（但分支仍存在，可通过
    :func:`unmerge_lora_from_base` 还原）。
    """
    if not HAS_TORCH:
        raise RuntimeError("训练需要安装 PyTorch。")
    merged = 0
    for _name, module in _iter_lora_layers(model):
        delta = _delta_weight(module)
        if strict_shape and delta.shape != module.base.weight.shape:
            raise RuntimeError(
                f"LoRA 增量形状 {tuple(delta.shape)} 与基础权重 "
                f"{tuple(module.base.weight.shape)} 不一致，拒绝合并。"
            )
        with torch.no_grad():
            module.base.weight.add_(delta)
        merged += 1
    return merged


def unmerge_lora_from_base(model) -> int:
    """撤销 :func:`merge_lora_to_base` 的效果（前提是 LoRA 参数未被清空）。"""
    if not HAS_TORCH:
        raise RuntimeError("训练需要安装 PyTorch。")
    unmerged = 0
    for _name, module in _iter_lora_layers(model):
        with torch.no_grad():
            module.base.weight.sub_(_delta_weight(module))
        unmerged += 1
    return unmerged


def lora_summary(model) -> dict:
    """汇总 LoRA 结构信息，用于日志/UI 展示。"""
    layers = list(_iter_lora_layers(model))
    lora_count, total_count = count_lora_parameters(model)
    ranks = {module.rank for _, module in layers}
    alphas = {module.alpha for _, module in layers}
    return {
        "layers": len(layers),
        "rank": sorted(ranks)[0] if len(ranks) == 1 else sorted(ranks),
        "alpha": sorted(alphas)[0] if len(alphas) == 1 else sorted(alphas),
        "lora_params": lora_count,
        "total_params": total_count,
        "ratio_percent": round(100.0 * lora_count / max(1, total_count), 4),
        "has_torch": HAS_TORCH,
        "runtime_version": getattr(lora_runtime, "__name__", "src.lora_runtime"),
    }
