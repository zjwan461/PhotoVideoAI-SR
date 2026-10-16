"""LoRA 工具：把 LoRA 注入 RealESRGAN、只导出 LoRA、推理时加载/卸载/切换。

与推理端的关系
--------------
* 训练前：:func:`prepare_lora_model` 构建基础网络 + 加载基础权重 + 注入 LoRA；
* 训练后：:func:`save_lora` 只导出 LoRA 参数到 ``lora_weights/``；
* 推理时：WebUI 下拉选择即由 :mod:`src.img_sr` 内部调用
  :func:`src.lora_runtime.load_lora_weights` 动态加载，可随时切换/卸载；
  若你想在自己脚本里对某个网络做同样的操作，用 :func:`attach_lora_for_inference`
  与 :func:`detach_lora`。

基础模型权重永远不会被写入或覆盖：本模块所有保存路径都限定在 ``lora_weights/``。
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import config
from src import img_sr, lora_runtime
from src.common import get_logger, scan_lora_weights

logger = get_logger("lora_utils")

try:
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False


class LoRAError(RuntimeError):
    """LoRA 工具相关的可读错误。"""


_SAFE_NAME = re.compile(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+")


def sanitize_lora_name(name: str) -> str:
    """把用户输入的 LoRA 名称清洗成安全文件名（保留中文、字母数字、-、_）。"""
    cleaned = _SAFE_NAME.sub("_", str(name or "").strip()).strip("_")
    return cleaned or f"lora_{time.strftime('%Y%m%d_%H%M%S')}"


def lora_output_path(name: str, *, suffix: str = ".pth", ensure_unique: bool = True) -> Path:
    """计算 LoRA 保存路径（限定在 ``lora_weights/`` 内）。"""
    config.LORA_DIR.mkdir(parents=True, exist_ok=True)
    stem = sanitize_lora_name(name)
    candidate = config.LORA_DIR / f"{stem}{suffix}"
    if ensure_unique:
        counter = 1
        while candidate.exists():
            candidate = config.LORA_DIR / f"{stem}_{counter}{suffix}"
            counter += 1
    return candidate


def resolve_lora_selection(selection: str | None) -> Path | None:
    """把 UI 下拉框选项解析成 lora_weights/ 内的绝对路径。"""
    if not selection or str(selection).startswith("不使用"):
        return None
    candidate = Path(str(selection))
    if not candidate.is_absolute():
        candidate = config.LORA_DIR / candidate
    try:
        candidate = candidate.resolve()
        candidate.relative_to(config.LORA_DIR.resolve())
    except (OSError, ValueError):
        return None
    return candidate if candidate.is_file() else None


# --------------------------------------------------------------------------- #
# 构建 / 注入
# --------------------------------------------------------------------------- #
def prepare_lora_model(
    model_name: str = config.DEFAULT_MODEL_NAME,
    *,
    rank: int = 8,
    alpha: float | None = None,
    device: str = "cuda:0",
    half: bool = False,
    auto_download: bool = False,
    network_factory=None,
):
    """构建「基础权重已加载 + LoRA 已注入」的可训练网络。

    Args:
        model_name: 基础模型文件名（决定结构与初始权重）。
        rank / alpha: LoRA 配置；alpha 默认等于 rank（scaling = 1）。
        device: 训练设备。
        half: 是否转半精度（训练建议 False，AMP 由 trainer 接管）。
        network_factory: 自定义结构工厂。默认走 :func:`src.img_sr.build_network`
            并加载对应的基础权重；**传入自定义工厂时会跳过基础权重加载**
            （自测或小模型实验用，例如 ``lambda _n: RRDBNet(num_block=1, ...)``）。

    Returns:
        ``(network, weights_path, info)``；自定义工厂时 ``weights_path`` 为 None。
    """
    if not HAS_TORCH:
        raise LoRAError("LoRA 微调需要安装 PyTorch（CUDA 版）。")

    if network_factory is not None:
        network = network_factory(model_name)
        weights: Path | None = None
        logger.warning("使用自定义网络工厂：跳过基础权重加载（仅用于自测/实验）。")
    else:
        network = img_sr.build_network(model_name)
        weights = img_sr.load_base_weights(network, model_name, auto_download=auto_download)

    lora_runtime.inject_lora(network, rank=int(rank), alpha=alpha, freeze=True)
    network = network.to(device)
    if half:
        network = network.half()
        logger.warning("训练使用半精度基础模型，建议改用 AMP 而非直接转 half。")

    from src.train import lora_network as ln

    info = ln.lora_summary(network)
    info["base_model"] = model_name if network_factory is None else "custom-factory"
    info["weights"] = str(weights) if weights else None
    info["device"] = device
    logger.info(
        "LoRA 训练模型就绪：%s（注入 %d 层，rank=%s，可训练 %d / %d 参数）",
        model_name, info["layers"], info["rank"], info["lora_params"], info["total_params"],
    )
    return network, weights, info


def attach_lora_for_inference(
    network,
    lora_path: str | Path,
    *,
    rank: int | None = None,
    alpha: float | None = None,
) -> dict:
    """给一个**干净的基础网络**注入并加载 LoRA（供自定义脚本使用）。"""
    if not HAS_TORCH:
        raise LoRAError("需要安装 PyTorch。")
    path = Path(lora_path)
    if not path.is_file():
        raise LoRAError(f"LoRA 权重不存在：{path}")
    if rank is None or alpha is None:
        try:
            payload = lora_runtime._read_checkpoint(path)  # noqa: SLF001
            inferred_rank, inferred_alpha = lora_runtime._peek_rank_from_checkpoint(payload)  # noqa: SLF001
        except Exception:  # noqa: BLE001
            inferred_rank, inferred_alpha = 8, 8.0
        rank = inferred_rank if rank is None else rank
        alpha = inferred_alpha if alpha is None else alpha
    lora_runtime.inject_lora(network, rank=int(rank), alpha=alpha, freeze=True)
    return lora_runtime.load_lora_weights(network, path)


def detach_lora(network) -> int:
    """移除网络中的 LoRA 包装，还原成原始结构。"""
    return lora_runtime.remove_lora(network)


# --------------------------------------------------------------------------- #
# 保存 / 读取
# --------------------------------------------------------------------------- #
def save_lora(
    model,
    name: str,
    *,
    rank: int | None = None,
    alpha: float | None = None,
    base_model: str | None = None,
    meta: dict | None = None,
    ensure_unique: bool = False,
) -> Path:
    """只导出 LoRA 参数（不导出基础模型）。"""
    target = lora_output_path(name, ensure_unique=ensure_unique)
    payload_meta = dict(meta or {})
    payload_meta.setdefault("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    return lora_runtime.save_lora_weights(
        model, target, rank=rank, alpha=alpha, base_model=base_model, meta=payload_meta
    )


def describe_lora(lora_path: str | Path) -> str:
    """读取 LoRA 文件信息（供 UI 展示）。"""
    return lora_runtime.describe_lora_file(lora_path)


def list_loras() -> list[dict]:
    """列出 ``lora_weights/`` 中的 LoRA 及其摘要信息。"""
    entries: list[dict] = []
    for name in scan_lora_weights():
        path = config.LORA_DIR / name
        try:
            size_mb = round(path.stat().st_size / (1024 ** 2), 2)
        except OSError:
            size_mb = None
        entries.append(
            {
                "name": name,
                "path": str(path),
                "size_mb": size_mb,
                "summary": describe_lora(path),
            }
        )
    return entries


def delete_lora(name: str) -> bool:
    """删除 ``lora_weights/`` 中的某个 LoRA（限定目录内）。"""
    path = resolve_lora_selection(name)
    if path is None:
        return False
    try:
        path.unlink()
        logger.info("已删除 LoRA：%s", path.name)
        return True
    except OSError as exc:
        logger.warning("删除 LoRA 失败：%s -> %s", path, exc)
        return False


__all__ = [
    "HAS_TORCH",
    "LoRAError",
    "sanitize_lora_name",
    "lora_output_path",
    "resolve_lora_selection",
    "prepare_lora_model",
    "attach_lora_for_inference",
    "detach_lora",
    "save_lora",
    "describe_lora",
    "list_loras",
    "delete_lora",
]
