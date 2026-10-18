"""老照片前置复原（P4-2，可选）：用 SwinIR 做**同尺寸**的去压缩块 / 去噪。

为什么需要这一层
----------------
RealESRGAN 是「放大」模型，对 JPEG 压缩块、胶片颗粒这类退化并不是专门设计的；
先把输入复原干净，超分能拿到更可靠的低频结构。**复原不改变尺寸**，放大仍交给超分，
两者职责分开，避免「又放大又复原」互相干扰。

实现选择
--------
* 网络结构直接复用 ``basicsr.archs.swinir_arch.SwinIR``（随 basicsr 一起装好），
  **不额外搬运第三方模型代码**，架构参数与官方推理脚本一致；
* 权重是可选的（``scripts/install.py --restore`` 下载），缺失时自动隐藏该功能；
* 大图必须分块推理：SwinIR 的显存随边长平方增长，4K 整图会直接 OOM。

与「超分」的关系
----------------
推荐链路：``前置复原（可选） → 超分 → 人脸修复（可选） → 清晰度增强``。
复原在**输入分辨率**上运行，代价很低（远小于超分本身）。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import config
from src.common import clear_cuda_cache, friendly_cuda_error, get_logger
from src.torchvision_compat import ensure_functional_tensor

logger = get_logger("restore")

try:  # numpy 为必需依赖，缺失说明环境没装好
    import numpy as np

    HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]
    HAS_NUMPY = False

try:
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

#: SwinIR 来自 basicsr；导入前必须先装 torchvision 兼容垫片（basicsr 会连带踩到）
try:
    ensure_functional_tensor()
    from basicsr.archs.swinir_arch import SwinIR

    HAS_SWINIR = True
    SWINIR_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - 依赖缺失
    SwinIR = None  # type: ignore[assignment]
    HAS_SWINIR = False
    SWINIR_IMPORT_ERROR = str(exc)


class RestoreError(RuntimeError):
    """前置复原过程中的可读错误。"""


def restore_label(key: str) -> str:
    """模型 key -> 展示名。"""
    entry = config.RESTORE_MODELS.get(key)
    return entry["label"] if entry else key


def build_network(key: str):
    """按登记的参数构建 SwinIR（不加载权重）。

    Args:
        key: ``config.RESTORE_MODELS`` 中的模型键，如 ``"jpeg"`` / ``"denoise"``。

    Raises:
        RestoreError: 依赖缺失或 key 未知。
    """
    if not HAS_SWINIR:
        raise RestoreError(
            "复原功能需要 basicsr 的 SwinIR 架构。\n"
            f"导入错误：{SWINIR_IMPORT_ERROR}"
        )
    entry = config.RESTORE_MODELS.get(key)
    if entry is None:
        raise RestoreError(f"未知的复原模型：{key}")
    try:
        return SwinIR(**entry["arch"])
    except Exception as exc:  # noqa: BLE001
        raise RestoreError(f"构建复原网络失败（{key}）：{exc}") from exc


def load_weights(network, key: str) -> Path:
    """把权重加载进网络，返回权重路径。"""
    path = config.restore_model_path(key)
    if not path.is_file():
        raise RestoreError(
            f"缺少复原模型权重：{path}\n"
            "执行 python scripts/install.py --restore 可自动下载。"
        )
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise RestoreError(f"读取复原权重失败（{path.name}）：{exc}") from exc

    state = (
        checkpoint.get("params")
        or checkpoint.get("params_ema")
        or checkpoint.get("state_dict")
        or checkpoint
    )
    if not isinstance(state, dict):
        raise RestoreError(f"复原权重的结构无法识别：{path.name}")
    try:
        missing, unexpected = network.load_state_dict(state, strict=False)
    except Exception as exc:  # noqa: BLE001
        raise RestoreError(f"复原权重与网络结构不匹配（{path.name}）：{exc}") from exc
    if missing or unexpected:
        # 不直接失败：不同版本的 checkpoint 可能多/少少量缓冲项
        logger.warning(
            "复原权重与网络存在差异（缺少 %d 项 / 多余 %d 项），已尽力加载",
            len(missing), len(unexpected),
        )
    return path


def _pad_to_window(tensor, window: int):
    """把 H/W 补到 ``window`` 的整数倍（SwinIR 的窗口划分要求）。

    basicsr 版的 ``SwinIR.forward`` **不自带 padding**（官方仓库是在外部用
    ``check_image_size`` 做的），所以这里必须自己补，否则窗口划分会直接报错。
    """
    height, width = tensor.shape[-2:]
    pad_h = (window - height % window) % window
    pad_w = (window - width % window) % window
    if not pad_h and not pad_w:
        return tensor, height, width
    # reflect 要求 pad < 对应边长，太小的图退回 replicate
    mode = "reflect" if (height > pad_h and width > pad_w) else "replicate"
    padded = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)
    return padded, height, width


def _forward_padded(model, patch, window: int):
    """补边 → 前向 → 裁回原尺寸。"""
    padded, height, width = _pad_to_window(patch, window)
    out = model(padded)
    return out[:, :, :height, :width]


def _tiled_forward(
    model,
    tensor,
    *,
    tile: int,
    window: int,
    overlap: int = 32,
    should_cancel: Callable[[], None] | None = None,
):
    """分块前向并做重叠区平均，控制显存同时避免拼缝。

    Args:
        window: SwinIR 的窗口大小。basicsr 版**没有把 window_size 存成模型属性**，
            必须由调用方从架构参数里传入，否则补边会补错。
    """
    window = max(1, int(window))
    height, width = tensor.shape[-2:]
    if tile <= 0 or (height <= tile and width <= tile):
        return _forward_padded(model, tensor, window)

    tile = int(tile)
    stride = max(1, tile - int(overlap))
    acc = torch.zeros_like(tensor)
    weight = torch.zeros((1, 1, height, width), device=tensor.device, dtype=tensor.dtype)
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            if should_cancel is not None:
                should_cancel()
            y0 = min(y, max(0, height - tile))
            x0 = min(x, max(0, width - tile))
            y1, x1 = min(y0 + tile, height), min(x0 + tile, width)
            acc[:, :, y0:y1, x0:x1] += _forward_padded(model, tensor[:, :, y0:y1, x0:x1], window)
            weight[:, :, y0:y1, x0:x1] += 1.0
    return acc / weight.clamp_min(1.0)


class RestoreEngine:
    """复原模型的缓存容器（同一时间只保留一个，切换时自动换）。"""

    def __init__(self) -> None:
        self.key: str | None = None
        self.model = None
        self.device: str = "cuda:0"
        self.half: bool = True
        self.window: int = 8
        #: "rgb" = 三通道直接复原；"y" = 只在 Y（亮度）通道复原后合并
        self.channels: str = "rgb"

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(
        self,
        key: str,
        *,
        device: str = "cuda:0",
        half: bool = True,
    ) -> "RestoreEngine":
        """加载（或复用）指定复原模型。"""
        if not HAS_TORCH:
            raise RestoreError("前置复原需要 PyTorch。")
        if not HAS_NUMPY:
            raise RestoreError("前置复原需要 numpy。")

        device = config.normalize_device(device)
        half = bool(half) and device.startswith("cuda")
        if self.ready and self.key == key and self.device == device and self.half == half:
            return self

        self.unload()
        started = time.time()
        network = build_network(key)
        path = load_weights(network, key)
        network.eval()
        # 注意：模型保持 fp32，前向时用 autocast 混精。
        # 直接 .half() 会让 SwinIR 的 attention 出现 Half/Float 类型冲突。
        try:
            network = network.to(device)
        except Exception as exc:  # noqa: BLE001
            raise RestoreError(f"复原模型移到 {device} 失败：{exc}") from exc

        self.key, self.model, self.device, self.half = key, network, device, half        # 窗口大小用于补边；basicsr 的 SwinIR 没有把它存成属性
        self.window = int(config.RESTORE_MODELS[key]["arch"].get("window_size", 8))
        # 灰度模型（in_chans=1）只在亮度通道复原，这是官方对彩色图的标准用法
        self.channels = str(config.RESTORE_MODELS[key].get("channels", "rgb"))
        logger.info(
            "复原模型加载完成：%s（%s，%s，%.2fs）",
            path.name, device, "fp16" if half else "fp32", time.time() - started,
        )
        return self

    def unload(self) -> None:
        if self.model is not None:
            self.model = None
            clear_cuda_cache()
        self.key = None

    def restore(
        self,
        image_rgb,
        *,
        progress: Callable[[float, str], None] | None = None,
        should_cancel: Callable[[], None] | None = None,
    ):
        """对一张 RGB 图做同尺寸复原，返回同尺寸 RGB 图。"""
        if not self.ready:
            raise RestoreError("复原模型尚未加载。")
        if image_rgb is None or getattr(image_rgb, "size", 0) == 0:
            raise RestoreError("输入图片为空。")

        has_alpha = image_rgb.ndim == 3 and image_rgb.shape[2] == 4
        rgb = image_rgb[:, :, :3] if has_alpha else image_rgb
        alpha = image_rgb[:, :, 3] if has_alpha else None

        # 灰度模型只在亮度通道上工作（官方对彩色图的标准做法）：
        # 转 YCrCb，复原 Y，再合并回去，色度保持原样。
        if self.channels == "y":
            import cv2

            ycbcr = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
            plane = ycbcr[:, :, 0]
        else:
            ycbcr = None
            plane = rgb

        source = plane[:, :, None] if plane.ndim == 2 else plane
        tensor = (
            torch.from_numpy(np.ascontiguousarray(source))
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .div_(255.0)
            .unsqueeze(0)
            .to(self.device)
        )
        if progress:
            progress(0.01, "前置复原中…")
        # 复原一律用 fp32：CAR 系列的 img_range=255 会让内部激活放大 255 倍，
        # 在 fp16 下会溢出成 NaN，随后转 uint8 就变成一张全黑图（不报错，极难发现）。
        try:
            with torch.inference_mode():
                output = _tiled_forward(
                    self.model,
                    tensor,
                    tile=int(config.RESTORE_TILE),
                    window=int(self.window),
                    should_cancel=should_cancel,
                )
        except Exception as exc:  # noqa: BLE001
            friendly = friendly_cuda_error(exc)
            if friendly:
                raise RestoreError(friendly) from exc
            raise RestoreError(
                "前置复原失败："
                f"{exc}\n若为显存不足，可用环境变量 PVSR_RESTORE_TILE 调小分块（如 256）。"
            ) from exc

        if not bool(torch.isfinite(output).all()):
            raise RestoreError(
                "前置复原输出出现 NaN/Inf（数值溢出）。\n"
                "请尝试用环境变量 PVSR_RESTORE_TILE 调小分块（如 256），或换用其它复原模型。"
            )

        array = (
            output.float()
            .squeeze(0)
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
        )
        if ycbcr is not None:
            # 单通道输出：写回亮度，色度沿用原图
            ycbcr[:, :, 0] = array[:, :, 0]
            import cv2

            array = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2RGB)
        if alpha is not None:
            array = np.dstack([array, alpha])
        return array


#: 进程级共享复原引擎
_ENGINE = RestoreEngine()


def get_engine() -> RestoreEngine:
    """获取共享复原引擎。"""
    return _ENGINE


def restore_image(
    image_rgb,
    key: str,
    *,
    device: str = "cuda:0",
    half: bool = True,
    progress: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], None] | None = None,
):
    """便捷入口：加载指定复原模型并复原一张图（同尺寸）。"""
    engine = get_engine().load(key, device=device, half=half)
    return engine.restore(image_rgb, progress=progress, should_cancel=should_cancel)


def probe(key: str, *, tolerance: float = 0.15) -> tuple[bool, str]:
    """自检：用常数输入探测模型是否接近「恒等」，返回 (是否正常, 说明)。

    why：这类复原模型的 ``img_range`` / ``in_chans`` 必须与训练时完全一致，
    取错时**不会报错**，只会输出整体偏移甚至全黑的图（本机就踩过两次）。
    复原模型对平坦输入应当近似恒等，因此「常数 0.5 进、≈0.5 出」是一个
    廉价而有效的正确性哨兵，安装脚本会在下载后自动跑一次。

    Args:
        key: 模型键。
        tolerance: 允许的平均偏移（0~1 归一化尺度）。
    """
    if not HAS_TORCH:
        return False, "PyTorch 不可用"
    try:
        network = build_network(key)
        load_weights(network, key)
    except RestoreError as exc:
        return False, str(exc).splitlines()[0]
    network.eval()

    channels = 1 if str(config.RESTORE_MODELS[key].get("channels")) == "y" else 3
    window = int(config.RESTORE_MODELS[key]["arch"].get("window_size", 8))
    probe_value = 0.5
    x = torch.full((1, channels, 64, 64), probe_value)
    try:
        with torch.inference_mode():
            y = _forward_padded(network, x, window)
    except Exception as exc:  # noqa: BLE001
        return False, f"前向失败：{exc}"

    values = y.float()
    if not bool(torch.isfinite(values).all()):
        return False, "输出出现 NaN/Inf（多为 fp16 溢出或配置不匹配）"
    mean = float(values.mean())
    delta = abs(mean - probe_value)
    text = f"常数 {probe_value:g} → {mean:.3f}（偏差 {delta:.3f}）"
    return delta <= tolerance, text


__all__ = [
    "HAS_SWINIR",
    "RestoreError",
    "RestoreEngine",
    "build_network",
    "load_weights",
    "restore_image",
    "restore_label",
    "probe",
    "get_engine",
]
