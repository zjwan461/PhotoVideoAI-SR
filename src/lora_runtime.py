"""LoRA 运行时：卷积层 LoRA 的注入 / 加载 / 卸载 / 保存。

设计要点
--------
* **手写实现，不依赖 peft**：只包装 ``nn.Conv2d``，对 GAN / 超分网络结构零假设。
* **主链路零污染**：不选择 LoRA 时模型保持原样，注入只发生在显式调用时。
* **格式与 P2 微调模块共用**：checkpoint 里只有 LoRA 参数、rank / alpha 与元信息，
  不含基础模型权重，因此文件通常只有几百 KB 到几 MB。

卷积 LoRA 形式：``y = Conv(x) + (alpha / rank) * B(A(x))``，
其中 A 为 1x1 卷积（in -> rank），B 为 kxk 卷积（rank -> out），二者复合等价于
对基础卷积核的低秩增量 ``ΔW = B·A``。初始化时 B 为零，注入后输出与原始模型完全一致。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from src.common import get_logger

logger = get_logger("lora")

try:  # torch 为可选前置，缺失时模块仍可被导入（UI 会给出提示）
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except Exception:  # pragma: no cover - 无 torch 环境
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    HAS_TORCH = False

#: 默认参与注入的模块名关键字；**空元组表示注入所有 Conv2d**（不依赖命名习惯）
DEFAULT_TARGET_KEYWORDS: tuple[str, ...] = ()

#: 不参与注入的模块名关键字（默认跳过输出层 conv_last，避免直接扰动 HR 图像统计）
DEFAULT_EXCLUDE_KEYWORDS: tuple[str, ...] = ("conv_last",)

_LORA_SUFFIXES = (".pth", ".pt", ".safetensors")


if HAS_TORCH:

    class LoRAConv2d(nn.Module):  # type: ignore[misc]
        """包装一个冻结的 ``nn.Conv2d``，并附加可训练的低秩增量分支。"""

        def __init__(self, base_conv, rank: int = 8, alpha: float = 8.0):
            super().__init__()
            if not isinstance(base_conv, nn.Conv2d):
                raise TypeError(f"LoRAConv2d 只支持 nn.Conv2d，收到 {type(base_conv)!r}")
            if base_conv.groups != 1:
                raise ValueError("分组卷积暂不支持 LoRA 注入。")

            self.rank = int(rank)
            self.alpha = float(alpha)
            self.scaling = self.alpha / max(1, self.rank)
            self.base = base_conv

            for parameter in self.base.parameters():
                parameter.requires_grad_(False)

            self.lora_A = nn.Conv2d(base_conv.in_channels, self.rank, kernel_size=1, bias=False)
            self.lora_B = nn.Conv2d(
                self.rank,
                base_conv.out_channels,
                kernel_size=base_conv.kernel_size,
                stride=base_conv.stride,
                padding=base_conv.padding,
                dilation=base_conv.dilation,
                bias=False,
            )
            self.reset_parameters()

        def reset_parameters(self) -> None:
            """A 用 Kaiming 初始化，B 置零 —— 注入瞬间不改变原有输出。"""
            nn.init.kaiming_normal_(self.lora_A.weight, a=0, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(self.lora_B.weight)

        def forward(self, x):
            base_out = self.base(x)
            delta = self.lora_B(self.lora_A(x)) * self.scaling
            return base_out + delta
else:  # pragma: no cover - 无 torch 环境

    class LoRAConv2d:  # type: ignore[no-redef]
        """torch 不可用时的占位实现。"""

        def __init__(self, *args, **kwargs):
            raise RuntimeError("LoRA 功能需要安装 PyTorch。")


def _iter_target_modules(model, keywords: Sequence[str], excludes: Sequence[str]):
    """遍历模型中匹配关键字的 (名称, 模块) 对。

    ``keywords`` 为空表示命中所有 ``nn.Conv2d``；``excludes`` 命中即跳过。
    分组卷积（``groups != 1``）会自动跳过，因为低秩增量与分组语义不兼容。
    """
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if module.groups != 1:
            continue
        lowered = name.lower()
        if keywords and not any(keyword in lowered for keyword in keywords):
            continue
        if any(blocked in lowered for blocked in excludes):
            continue
        yield name, module


def _set_submodule(root, dotted_name: str, new_module) -> None:
    """按点分路径替换子模块。"""
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _get_submodule(root, dotted_name: str):
    module = root
    for part in dotted_name.split("."):
        module = getattr(module, part)
    return module


def freeze_base_parameters(model) -> int:
    """冻结模型全部原参数，只保留 LoRA 参数可训练。返回可训练参数个数。"""
    if not HAS_TORCH:
        raise RuntimeError("LoRA 功能需要安装 PyTorch。")
    trainable = 0
    for name, parameter in model.named_parameters():
        is_lora = ".lora_A." in name or ".lora_B." in name or name.startswith("lora_")
        parameter.requires_grad_(is_lora)
        if is_lora:
            trainable += parameter.numel()
    return trainable


def inject_lora(
    model,
    rank: int = 8,
    alpha: float | None = None,
    keywords: Iterable[str] | None = None,
    excludes: Iterable[str] | None = None,
    *,
    freeze: bool = True,
) -> list[str]:
    """把卷积 LoRA 注入模型。

    Args:
        model: 待注入的 ``nn.Module``（通常是 RealESRGAN 的网络部分）。
        rank: LoRA 秩，越小参数越少。
        alpha: 缩放系数，默认等于 rank（scaling = 1）。
        keywords: 命中这些关键字的 Conv2d 才会被注入；None 或空表示注入所有卷积层。
        excludes: 命中即跳过的关键字（默认跳过 ``conv_last``）。
        freeze: 是否冻结基础权重（推理时可设 False 以保持原样）。

    Returns:
        被注入的模块名列表（可用于保存时的键前缀）。
    """
    if not HAS_TORCH:
        raise RuntimeError("LoRA 功能需要安装 PyTorch。")

    keyword_list = tuple(keywords) if keywords else DEFAULT_TARGET_KEYWORDS
    exclude_list = tuple(excludes) if excludes else DEFAULT_EXCLUDE_KEYWORDS
    alpha_value = float(alpha) if alpha is not None else float(rank)

    injected: list[str] = []
    for name, module in list(_iter_target_modules(model, keyword_list, exclude_list)):
        wrapper = LoRAConv2d(module, rank=rank, alpha=alpha_value)
        _set_submodule(model, name, wrapper)
        injected.append(name)

    if freeze:
        freeze_base_parameters(model)

    if not injected:
        raise RuntimeError(
            "没有匹配到可注入的卷积层，请检查 keywords/excludes 配置是否与模型结构相符。"
        )
    logger.info("LoRA 注入完成：rank=%s alpha=%s 层数=%d", rank, alpha_value, len(injected))
    return injected


def remove_lora(model) -> int:
    """移除模型中的全部 LoRA 包装，还原为原始卷积。返回还原的层数。"""
    if not HAS_TORCH:
        return 0
    restored = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRAConv2d):
            _set_submodule(model, name, module.base)
            restored += 1
    return restored


def lora_state_dict(model, prefix: str = "") -> dict:
    """抽取模型中的 LoRA 参数（键名去掉 DataParallel 前缀与包装层，便于跨结构加载）。"""
    if not HAS_TORCH:
        raise RuntimeError("LoRA 功能需要安装 PyTorch。")
    state: dict = {}
    for name, parameter in model.named_parameters():
        if ".lora_A." not in name and ".lora_B." not in name:
            continue
        key = name.replace(".base.", ".")
        if key.startswith("module."):
            key = key[len("module."):]
        state[(prefix + key) if prefix else key] = parameter.detach().cpu().clone()
    return state


def count_lora_parameters(model) -> tuple[int, int]:
    """返回 ``(LoRA 参数量, 模型总参数量)``。"""
    if not HAS_TORCH:
        return 0, 0
    lora_params = sum(
        p.numel() for n, p in model.named_parameters() if ".lora_A." in n or ".lora_B." in n
    )
    total = sum(p.numel() for p in model.parameters())
    return lora_params, total


def _peek_rank_from_checkpoint(payload: dict) -> tuple[int, float]:
    """从 checkpoint 的 state_dict 形状推断 rank / alpha。"""
    state = payload.get("lora_state_dict") or payload.get("state_dict") or payload
    rank = payload.get("rank")
    alpha = payload.get("alpha")
    if rank is None:
        for key, value in state.items():
            if key.endswith("lora_A.weight") and hasattr(value, "shape"):
                rank = int(value.shape[0])
                break
    rank = int(rank) if rank else 8
    alpha = float(alpha) if alpha else float(rank)
    return rank, alpha


def _read_checkpoint(checkpoint_path: Path) -> dict:
    """读取 LoRA 文件；.safetensors 走 safetensors，其余走 torch.load。"""
    if checkpoint_path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file

            return {"lora_state_dict": load_file(str(checkpoint_path))}
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"读取 LoRA 文件失败（safetensors）：{exc}") from exc
    try:
        return torch.load(str(checkpoint_path), map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"读取 LoRA 文件失败：{exc}") from exc


def load_lora_weights(
    model,
    checkpoint_path: str | Path,
    *,
    strict: bool = False,
    min_match_ratio: float = 0.5,
) -> dict:
    """把 LoRA 权重加载进**已注入** LoRA 的模型。

    Args:
        model: 已经通过 :func:`inject_lora` 包装过的模型。
        checkpoint_path: ``lora_weights/*.pth``。
        strict: 是否要求所有键完全匹配。
        min_match_ratio: 允许的最低匹配比例；低于该比例视为“LoRA 与当前模型不匹配”并报错，
            避免静默地只加载了零星几个张量却让用户以为已经生效。

    Returns:
        dict: ``{"loaded": n, "missing": [...], "unexpected": [...]}``
    """
    if not HAS_TORCH:
        raise RuntimeError("LoRA 功能需要安装 PyTorch。")

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"LoRA 权重不存在：{path}")

    payload = _read_checkpoint(path)
    state = payload.get("lora_state_dict") or payload.get("state_dict") or payload
    if not isinstance(state, dict) or not state:
        raise RuntimeError("LoRA 文件内容为空或格式不正确。")

    target = model.module if hasattr(model, "module") else model
    current = dict(target.named_parameters())
    loaded, missing, unexpected = 0, [], []

    for key, value in state.items():
        clean = key[len("module."):] if key.startswith("module.") else key
        parameter = current.get(clean)
        if parameter is None:
            unexpected.append(key)
            continue
        try:
            with torch.no_grad():
                parameter.copy_(value.to(parameter.device, dtype=parameter.dtype))
            loaded += 1
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{key} ({exc})")

    if strict and (missing or unexpected):
        raise RuntimeError(
            f"LoRA 权重与模型结构不匹配：缺失 {len(missing)} 项，多余 {len(unexpected)} 项。"
        )
    if loaded == 0:
        raise RuntimeError(
            "LoRA 权重未能匹配任何参数。请确认该 LoRA 与当前基础模型结构一致"
            f"（文件 {path.name}，共 {len(state)} 个张量）。"
        )
    if loaded < len(state) * max(0.0, min(1.0, min_match_ratio)):
        raise RuntimeError(
            f"LoRA 权重与当前基础模型不匹配：仅匹配 {loaded}/{len(state)} 个张量"
            f"（多余 {len(unexpected)} 项），已拒绝加载以免产生错误效果。\n"
            "请确认该 LoRA 是用同一个基础模型（如 RealESRGAN_x4plus.pth）训练的。"
        )

    logger.info("LoRA 加载完成：%s（匹配 %d/%d 个张量）", path.name, loaded, len(state))
    return {"loaded": loaded, "missing": missing, "unexpected": unexpected}


def save_lora_weights(
    model,
    output_path: str | Path,
    *,
    rank: int | None = None,
    alpha: float | None = None,
    base_model: str | None = None,
    meta: dict | None = None,
) -> Path:
    """只导出 LoRA 参数（绝不包含基础模型权重）。"""
    if not HAS_TORCH:
        raise RuntimeError("LoRA 功能需要安装 PyTorch。")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    state = lora_state_dict(model)
    if not state:
        raise RuntimeError("模型中没有任何 LoRA 参数，无法保存。")

    if rank is None:
        rank, _ = _peek_rank_from_checkpoint({"lora_state_dict": state})
    if alpha is None:
        alpha = float(rank)

    payload = {
        "format": "photovideoai-sr-lora",
        "format_version": 1,
        "rank": int(rank),
        "alpha": float(alpha),
        "base_model": base_model,
        "meta": meta or {},
        "lora_state_dict": state,
    }
    torch.save(payload, str(target))
    size_mb = target.stat().st_size / (1024**2)
    logger.info("LoRA 已保存 -> %s (%.2f MB, %d 个张量)", target, size_mb, len(state))
    return target


def describe_lora_file(checkpoint_path: str | Path) -> str:
    """读取 LoRA 文件的基本信息，用于 UI 展示。"""
    path = Path(checkpoint_path)
    if not path.is_file():
        return f"文件不存在：{path}"
    if not HAS_TORCH:
        return f"{path.name}（未安装 PyTorch，无法读取元信息）"
    try:
        payload = _read_checkpoint(path)
    except Exception as exc:  # noqa: BLE001
        return f"{path.name}（读取失败：{exc}）"
    state = payload.get("lora_state_dict") or payload.get("state_dict") or payload
    rank, alpha = _peek_rank_from_checkpoint(payload if isinstance(payload, dict) else {})
    tensors = len(state) if isinstance(state, dict) else 0
    base = payload.get("base_model") if isinstance(payload, dict) else None
    size_mb = path.stat().st_size / (1024**2)
    return (
        f"{path.name} | rank={rank} alpha={alpha:g} | 张量数={tensors} | "
        f"大小={size_mb:.2f} MB" + (f" | 基础模型={base}" if base else "")
    )


__all__ = [
    "HAS_TORCH",
    "LoRAConv2d",
    "inject_lora",
    "remove_lora",
    "load_lora_weights",
    "save_lora_weights",
    "lora_state_dict",
    "freeze_base_parameters",
    "count_lora_parameters",
    "describe_lora_file",
    "DEFAULT_TARGET_KEYWORDS",
    "DEFAULT_EXCLUDE_KEYWORDS",
]
