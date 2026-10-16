"""RealESRGAN 图片超分推理，支持可选 LoRA 注入与目标分辨率缩放。

对外主入口：
* :func:`upscale_image` —— 单张图片超分（RGB numpy 进、RGB numpy 出）
* :func:`upscale_folder` —— 批量文件夹超分（P1）
* :class:`RealESRGANEngine` —— 带缓存的模型管理器

异常约定：所有失败都抛出 :class:`ImageSRError`（或其子类），
message 已是面向用户的可读文案，UI 直接展示即可。
"""

from __future__ import annotations

import gc
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

import config
from src import lora_runtime
from src.common import (
    clear_cuda_cache,
    friendly_cuda_error,
    get_free_vram_gb,
    get_logger,
    human_size,
    list_image_files,
    resolve_lora_path,
    unique_output_path,
)

logger = get_logger("img_sr")

# --------------------------------------------------------------------------- #
# 可选依赖
# --------------------------------------------------------------------------- #
try:
    import numpy as np

    HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]
    HAS_NUMPY = False

try:
    import cv2

    HAS_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    HAS_CV2 = False

try:
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    HAS_REALESRGAN = True
    REALESRGAN_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - 依赖缺失
    RRDBNet = None  # type: ignore[assignment]
    RealESRGANer = None  # type: ignore[assignment]
    HAS_REALESRGAN = False
    REALESRGAN_IMPORT_ERROR = str(exc)


class ImageSRError(RuntimeError):
    """图片超分过程中的可读错误。"""


class ModelNotFoundError(ImageSRError):
    """基础模型权重缺失。"""


# --------------------------------------------------------------------------- #
# 模型文件管理
# --------------------------------------------------------------------------- #
def model_path(model_name: str) -> Path:
    """返回基础模型权重的本地路径。"""
    return config.MODEL_DIR / model_name


def available_base_models() -> list[str]:
    """列出 model_weights/ 下已有的基础模型（缺失时回退到默认名）。"""
    names = [
        name
        for name in config.MODEL_REGISTRY
        if (config.MODEL_DIR / name).is_file()
    ]
    if not names:
        return [config.DEFAULT_MODEL_NAME]
    return names


def download_model(
    model_name: str,
    *,
    progress: Callable[[float, str], None] | None = None,
    timeout: int = 60,
) -> Path:
    """下载基础模型权重到 model_weights/（P1：缺失时自动下载）。

    自动下载仅在我们知道官方地址时进行；未知文件名直接报错。
    """
    entry = config.MODEL_REGISTRY.get(model_name)
    if entry is None:
        raise ModelNotFoundError(
            f"未知的模型文件名 {model_name}。请手动放入 {config.MODEL_DIR} 目录，"
            "或在 config.MODEL_REGISTRY 中补充下载地址。"
        )

    target = model_path(model_name)
    if target.is_file():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    url = entry["url"]
    logger.info("开始下载基础模型：%s <- %s", model_name, url)

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            with open(partial, "wb") as handle:
                while True:
                    chunk = response.read(1024 * 512)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if progress and total:
                        progress(
                            min(0.99, downloaded / total),
                            f"下载模型 {model_name}：{human_size(downloaded)} / {human_size(total)}",
                        )
    except Exception as exc:  # noqa: BLE001
        partial.unlink(missing_ok=True)
        raise ModelNotFoundError(
            f"模型自动下载失败：{exc}\n"
            f"请手动下载 {url}\n并保存为 {target}"
        ) from exc

    partial.replace(target)
    if progress:
        progress(1.0, f"模型下载完成：{target.name}")
    logger.info("模型下载完成 -> %s (%s)", target, human_size(target.stat().st_size))
    return target


def ensure_base_model(
    model_name: str,
    *,
    auto_download: bool = False,
    progress: Callable[[float, str], None] | None = None,
) -> Path:
    """确保基础模型存在；缺失时按需自动下载，否则给出可读错误。"""
    target = model_path(model_name)
    if target.is_file():
        return target
    if auto_download:
        return download_model(model_name, progress=progress)
    url = config.MODEL_REGISTRY.get(model_name, {}).get("url", "见 README「模型下载」章节")
    raise ModelNotFoundError(
        f"未找到基础模型权重：{target}\n"
        f"下载地址：{url}\n"
        "也可以勾选「自动下载缺失模型」，或手动放入 model_weights/ 目录。"
    )


# --------------------------------------------------------------------------- #
# 网络构建（推理与训练共用）
# --------------------------------------------------------------------------- #
def model_scale(model_name: str) -> int:
    """返回基础模型固有的放大倍数。"""
    return config.MODEL_SCALE.get(model_name, 4)


def model_num_block(model_name: str) -> int:
    """按模型文件名推断 RRDB 块数量（动漫版为 6B）。"""
    return 6 if "anime" in str(model_name).lower() else 23


def build_network(model_name: str = config.DEFAULT_MODEL_NAME, *, num_block: int | None = None):
    """构建与指定基础模型匹配的 RRDBNet 结构（**不加载权重**）。

    推理端先建结构再让 RealESRGANer 加载权重；训练端（P2）复用同一函数，
    保证「训练时的结构」与「推理时的结构」始终一致。

    Args:
        model_name: 基础模型文件名，用于推断 scale 与块数。
        num_block: 手动覆盖块数（自测/小模型实验用）。
    """
    if not HAS_REALESRGAN:
        raise ImageSRError(
            "未安装 RealESRGAN 推理依赖（realesrgan / basicsr）。\n"
            "请执行：pip install -r requirements.txt\n"
            f"导入错误：{REALESRGAN_IMPORT_ERROR or '未知'}"
        )
    scale = model_scale(model_name)
    blocks = model_num_block(model_name) if num_block is None else int(num_block)
    try:
        return RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=blocks,
            num_grow_ch=32,
            scale=scale,
        )
    except Exception as exc:  # noqa: BLE001
        raise ImageSRError(f"构建 RealESRGAN 网络失败：{exc}") from exc


def load_base_weights(network, model_name: str = config.DEFAULT_MODEL_NAME, *, auto_download: bool = False) -> Path:
    """把基础权重加载进给定网络（训练起始点）。

    兼容 ``params`` / ``params_ema`` / 裸 state_dict 三种常见的权重格式。
    """
    if not HAS_TORCH:
        raise ImageSRError("未安装 PyTorch，请先安装 CUDA 版 PyTorch。")
    weights = ensure_base_model(model_name, auto_download=auto_download)
    try:
        payload = torch.load(str(weights), map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise ImageSRError(f"读取基础权重失败：{weights.name} -> {exc}") from exc

    if isinstance(payload, dict):
        for key in ("params_ema", "params", "state_dict"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise ImageSRError(f"基础权重格式不支持：{weights.name}")

    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in payload.items()}
    try:
        network.load_state_dict(state, strict=True)
    except Exception as exc:  # noqa: BLE001
        raise ImageSRError(
            f"基础权重与网络结构不匹配：{weights.name}\n{exc}\n"
            "请确认模型文件名与结构对应（例如动漫版对应 *_anime_6B.pth）。"
        ) from exc
    logger.info("基础权重已加载：%s", weights.name)
    return weights


# --------------------------------------------------------------------------- #
# 图像工具
# --------------------------------------------------------------------------- #
def _import_numpy():
    if not HAS_NUMPY or not HAS_CV2:
        raise ImageSRError(
            "缺少图像处理依赖（numpy / opencv-python）。请先执行：pip install -r requirements.txt"
        )
    return np


def read_image(path: str | Path):
    """读取图片为 RGB numpy 数组；损坏或格式不支持时抛出可读错误。"""
    _import_numpy()
    source = Path(path)
    if not source.is_file():
        raise ImageSRError(f"图片不存在：{source}")
    try:
        buffer = np.fromfile(str(source), dtype=np.uint8)  # 兼容中文/空格路径
        image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    except Exception as exc:  # noqa: BLE001
        raise ImageSRError(f"读取图片失败（文件可能已损坏）：{source.name} -> {exc}") from exc
    if image is None or image.size == 0:
        raise ImageSRError(f"无法解码图片（文件损坏或格式不支持）：{source.name}")
    return bgr_to_rgb(image)


def bgr_to_rgb(image):
    """OpenCV 的 BGR(A) -> RGB(A)。"""
    _import_numpy()
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image):
    """RGB(A) -> OpenCV 的 BGR(A)。"""
    _import_numpy()
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def write_image(path: str | Path, image_rgb) -> Path:
    """把 RGB numpy 数组写盘（根据后缀选择编码参数）。"""
    _import_numpy()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    bgr = rgb_to_bgr(image_rgb)
    params: list[int] = []
    suffix = target.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    elif suffix == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    try:
        ok, buffer = cv2.imencode(suffix or ".png", bgr, params)
        if not ok:
            raise ImageSRError(f"图片编码失败：{target.name}")
        buffer.tofile(str(target))
    except ImageSRError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ImageSRError(f"写入图片失败：{target} -> {exc}") from exc
    return target


def resize_to_long_edge(image, long_edge: int):
    """用 Lanczos 把图片等比缩放到指定长边。

    Args:
        long_edge: 目标长边像素；<=0 表示不缩放（原样返回）。
    """
    _import_numpy()
    if not long_edge or long_edge <= 0:
        return image
    height, width = image.shape[:2]
    current_long = max(height, width)
    if current_long == long_edge:
        return image
    scale = long_edge / float(current_long)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)


def resolve_target_long_edge(preset_name: str | None) -> int:
    """把分辨率预设名解析为长边像素；「原始比例」返回 0。"""
    width, height = config.RESOLUTION_PRESETS.get(
        preset_name or config.DEFAULT_RESOLUTION_PRESET, (0, 0)
    )
    return max(width, height)


def _auto_tile(device_index: int) -> int:
    """根据空闲显存自动选择分块大小（0 表示不分块）。"""
    free_gb = get_free_vram_gb(device_index)
    if free_gb is None:
        return 0
    if free_gb < 4:
        return 200
    if free_gb < 6:
        return 300
    if free_gb < 10:
        return 400
    return 0


# --------------------------------------------------------------------------- #
# 引擎
# --------------------------------------------------------------------------- #
class RealESRGANEngine:
    """RealESRGAN 推理引擎（进程内缓存，避免重复加载权重）。

    同一个实例可服务图片与视频逐帧推理；切换模型/LoRA 时会重建。
    """

    def __init__(self) -> None:
        self.model_name: str | None = None
        self.device: str = "cuda:0"
        self.half: bool = True
        self.tile: int = 0
        self.lora_name: str | None = None
        self.lora_info: dict | None = None
        self.upsampler = None
        self.scale: int = 4
        self._network = None

    # -- 生命周期 -------------------------------------------------------- #
    def load(
        self,
        *,
        model_name: str = config.DEFAULT_MODEL_NAME,
        device: str = "cuda:0",
        half: bool | None = None,
        tile: int | None = None,
        lora_selection: str | None = None,
        auto_download: bool = False,
        progress: Callable[[float, str], None] | None = None,
    ) -> RealESRGANEngine:
        """加载（或复用）基础模型，并按需注入 LoRA。"""
        if not HAS_REALESRGAN:
            raise ImageSRError(
                "未安装 RealESRGAN 推理依赖（realesrgan / basicsr）。\n"
                "请执行：pip install -r requirements.txt\n"
                f"导入错误：{REALESRGAN_IMPORT_ERROR or '未知'}"
            )
        if not HAS_TORCH:
            raise ImageSRError("未安装 PyTorch，请先安装 CUDA 版 PyTorch。")

        half = config.HALF_PRECISION if half is None else bool(half)
        device_index = config.gpu_index_from_option(device)
        resolved_tile = _auto_tile(device_index) if tile is None else int(tile)

        lora_path = resolve_lora_path(lora_selection)
        lora_key = str(lora_path) if lora_path else None

        unchanged = (
            self.upsampler is not None
            and self.model_name == model_name
            and self.device == device
            and self.half == half
            and self.tile == resolved_tile
            and self.lora_name == lora_key
        )
        if unchanged:
            logger.info("复用已加载模型：%s (%s)", model_name, device)
            return self

        self.unload()

        if progress:
            progress(0.02, "检查模型权重…")
        weights = ensure_base_model(model_name, auto_download=auto_download, progress=progress)

        if progress:
            progress(0.06, f"构建网络并加载权重：{weights.name}")
        self.scale = model_scale(model_name)
        network = build_network(model_name)

        # 注意：必须先用原始结构加载基础权重，再注入 LoRA。
        # 反过来的话，state_dict 的键名会多一层 `.base.`，严格加载会失败。
        try:
            upsampler = self._build_upsampler(network, weights, device, half, resolved_tile)
        except Exception as exc:  # noqa: BLE001
            friendly = friendly_cuda_error(exc)
            raise ImageSRError(friendly or f"加载模型失败：{exc}") from exc

        self.upsampler = upsampler
        self._network = network
        self.model_name = model_name
        self.device = device
        self.half = half
        self.tile = resolved_tile
        self.lora_name = lora_key
        self.lora_info = None

        if lora_path is not None:
            if not lora_runtime.HAS_TORCH:
                self.unload()
                raise ImageSRError("LoRA 需要 PyTorch，请先安装。")
            try:
                rank, alpha = self._peek_lora_rank(lora_path)
                lora_runtime.inject_lora(upsampler.model, rank=rank, alpha=alpha, freeze=True)
                self.lora_info = lora_runtime.load_lora_weights(upsampler.model, lora_path)
            except Exception as exc:  # noqa: BLE001
                self.unload()
                raise ImageSRError(f"加载 LoRA 失败：{exc}") from exc
            lora_params, total_params = lora_runtime.count_lora_parameters(upsampler.model)
            logger.info(
                "LoRA 生效：%s（LoRA 参数 %d / 总计 %d）",
                Path(lora_key).name if lora_key else "-",
                lora_params,
                total_params,
            )

        if progress:
            progress(0.10, "模型就绪")
        logger.info(
            "模型加载完成：%s device=%s half=%s tile=%s lora=%s",
            model_name, device, half, resolved_tile or "auto(off)", Path(lora_key).name if lora_key else "无",
        )
        return self

    @staticmethod
    def _peek_lora_rank(lora_path: Path) -> tuple[int, float]:
        """读取 LoRA 文件里的 rank / alpha，用于构造匹配的结构。"""
        try:
            payload = lora_runtime._read_checkpoint(lora_path)  # noqa: SLF001
            return lora_runtime._peek_rank_from_checkpoint(payload)  # noqa: SLF001
        except Exception:  # noqa: BLE001
            logger.warning("无法读取 LoRA 元信息，按默认 rank=8 处理：%s", lora_path.name)
            return 8, 8.0

    @staticmethod
    def _build_upsampler(network, weights: Path, device: str, half: bool, tile: int):
        """兼容不同版本 realesrgan 的构造参数差异。"""
        device_index = config.gpu_index_from_option(device)
        common_kwargs = {
            "scale": config.MODEL_SCALE.get(weights.name, 4),
            "model_path": str(weights),
            "model": network,
            "tile": int(tile),
            "tile_pad": config.TILE_PAD,
            "pre_pad": 0,
            "half": bool(half),
            "device": device,
        }
        try:
            return RealESRGANer(**common_kwargs)
        except TypeError:
            # 老版本签名使用 gpu_id
            common_kwargs.pop("device", None)
            common_kwargs["gpu_id"] = device_index
            return RealESRGANer(**common_kwargs)

    def unload(self) -> None:
        """释放模型与显存。"""
        if self.upsampler is not None:
            try:
                del self.upsampler
            except Exception:  # noqa: BLE001
                pass
        self.upsampler = None
        self._network = None
        self.model_name = None
        self.lora_name = None
        self.lora_info = None
        gc.collect()
        clear_cuda_cache()

    @property
    def ready(self) -> bool:
        return self.upsampler is not None

    # -- 推理 ------------------------------------------------------------ #
    def upscale(
        self,
        image_rgb,
        *,
        outscale: float = 4.0,
        denoise_strength: float = 0.5,
        target_long_edge: int = 0,
        face_restore: bool = False,
        face_restorer=None,
    ):
        """对单张 RGB 图片做超分，并按需 Lanczos 缩放到目标长边。"""
        _import_numpy()
        if not self.ready:
            raise ImageSRError("模型尚未加载，请先调用 load()。")
        if image_rgb is None or getattr(image_rgb, "size", 0) == 0:
            raise ImageSRError("输入图片为空。")

        has_alpha = image_rgb.ndim == 3 and image_rgb.shape[2] == 4
        if has_alpha:
            rgb = image_rgb[:, :, :3]
            alpha = image_rgb[:, :, 3]
        else:
            rgb = image_rgb
            alpha = None

        bgr = rgb_to_bgr(rgb)
        try:
            output, _ = self.upsampler.enhance(
                bgr, outscale=float(outscale), alpha_upsampler="realesrgan"
            )
        except Exception as exc:  # noqa: BLE001
            friendly = friendly_cuda_error(exc)
            if friendly:
                raise ImageSRError(friendly) from exc
            raise ImageSRError(f"超分推理失败：{exc}") from exc

        result = bgr_to_rgb(output)

        if face_restore and face_restorer is not None:
            try:
                result = face_restorer.restore(result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("人脸修复失败，已跳过：%s", exc)

        if alpha is not None:
            alpha_up = cv2.resize(
                alpha,
                (result.shape[1], result.shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )
            result = np.dstack([result, alpha_up])

        if target_long_edge:
            if result.shape[2] == 4:
                rgb_part = resize_to_long_edge(result[:, :, :3], target_long_edge)
                alpha_part = cv2.resize(
                    result[:, :, 3], (rgb_part.shape[1], rgb_part.shape[0]), interpolation=cv2.INTER_LANCZOS4
                )
                result = np.dstack([rgb_part, alpha_part])
            else:
                result = resize_to_long_edge(result, target_long_edge)

        return result


#: 进程级共享引擎（图片 Tab 与视频 Tab 共用，避免重复加载）
_ENGINE = RealESRGANEngine()


def get_engine() -> RealESRGANEngine:
    """获取共享引擎实例。"""
    return _ENGINE


# --------------------------------------------------------------------------- #
# 人脸修复（可选）
# --------------------------------------------------------------------------- #
class FaceRestorer:
    """GFPGAN 封装（可选依赖，缺库时 UI 会自动隐藏开关）。"""

    def __init__(self, device: str = "cuda:0", upscale: int = 2):
        try:
            from gfpgan import GFPGANer
        except Exception as exc:  # noqa: BLE001
            raise ImageSRError(
                "未安装 GFPGAN。可选依赖安装方式：pip install gfpgan\n"
                f"导入错误：{exc}"
            ) from exc
        weights = config.MODEL_DIR / "GFPGANv1.4.pth"
        if not weights.is_file():
            raise ImageSRError(
                f"缺少人脸修复权重：{weights}\n下载地址：{config.GFPGAN_MODEL_URL}"
            )
        self.upscale = upscale
        self.restorer = GFPGANer(
            model_path=str(weights),
            upscale=upscale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=device,
        )
        self.device = device

    def restore(self, image_rgb):
        """对整图做人脸修复，返回 RGB 数组。"""
        _import_numpy()
        try:
            _cropped, _restored, restored_img = self.restorer.enhance(
                rgb_to_bgr(image_rgb),
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=0.5,
            )
        except Exception as exc:  # noqa: BLE001
            raise ImageSRError(f"人脸修复失败：{exc}") from exc
        if restored_img is None:
            return image_rgb
        return bgr_to_rgb(restored_img)


# --------------------------------------------------------------------------- #
# 对外主入口
# --------------------------------------------------------------------------- #
def estimate_required_vram_gb(target_long_edge: int = 0, outscale: float = 4.0) -> float:
    """粗略估算图片超分所需显存（GiB）。"""
    base = 2.5 + 0.8 * max(1.0, float(outscale))
    if target_long_edge >= 3840:
        base += 3.0
    elif target_long_edge >= 2560:
        base += 1.5
    return round(base, 1)


def upscale_image(
    image_rgb,
    *,
    preset_name: str | None = None,
    device: str = "cuda:0",
    model_name: str = config.DEFAULT_MODEL_NAME,
    lora_selection: str | None = None,
    half: bool | None = None,
    outscale: float = 4.0,
    tile: int | None = None,
    auto_download: bool = True,
    face_restore: bool = False,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[object, dict]:
    """单张图片超分（numpy RGB 进、numpy RGB 出）。

    Returns:
        ``(result_rgb, info)``，info 含耗时、输入输出尺寸、模型与 LoRA 信息。
    """
    _import_numpy()
    started = time.time()
    target_long_edge = resolve_target_long_edge(preset_name)

    if progress:
        progress(0.01, "准备中…")

    engine = get_engine().load(
        model_name=model_name,
        device=device,
        half=half,
        tile=tile,
        lora_selection=lora_selection,
        auto_download=auto_download,
        progress=progress,
    )

    face_restorer = None
    if face_restore:
        if progress:
            progress(0.12, "初始化人脸修复模型…")
        face_restorer = FaceRestorer(device=device)

    if progress:
        progress(0.15, "超分推理中…")
    result = engine.upscale(
        image_rgb,
        outscale=outscale,
        target_long_edge=target_long_edge,
        face_restore=face_restore,
        face_restorer=face_restorer,
    )
    if progress:
        progress(1.0, "完成")

    elapsed = time.time() - started
    info = {
        "elapsed": elapsed,
        "input_size": f"{image_rgb.shape[1]}x{image_rgb.shape[0]}",
        "output_size": f"{result.shape[1]}x{result.shape[0]}",
        "model": model_name,
        "lora": Path(engine.lora_name).name if engine.lora_name else "不使用 LoRA",
        "device": device,
        "free_vram_gb": get_free_vram_gb(config.gpu_index_from_option(device)),
    }
    logger.info(
        "图片超分完成：%s -> %s，耗时 %.2fs（模型=%s，LoRA=%s）",
        info["input_size"], info["output_size"], elapsed, model_name, info["lora"],
    )
    return result, info


def upscale_path(
    input_path: str | Path,
    *,
    output_dir: Path | None = None,
    image_format: str = config.DEFAULT_IMAGE_FORMAT,
    **kwargs,
) -> tuple[Path, dict]:
    """按路径读取图片 -> 超分 -> 按指定格式写盘。"""
    image = read_image(input_path)
    result, info = upscale_image(image, **kwargs)
    suffix = ".jpg" if str(image_format).upper() == "JPG" else ".png"
    target = unique_output_path(
        Path(input_path).stem + "_sr", suffix, subdir=output_dir.name if output_dir else None
    )
    if output_dir is not None:
        target = Path(output_dir) / target.name
    write_image(target, result)
    info["output_path"] = str(target)
    return target, info


def upscale_folder(
    input_dir: str | Path,
    *,
    output_subdir: str = "batch",
    image_format: str = config.DEFAULT_IMAGE_FORMAT,
    recursive: bool = True,
    progress: Callable[[float, str], None] | None = None,
    **kwargs,
) -> tuple[list[Path], list[str], dict]:
    """批量处理文件夹内的所有图片（P1）。

    Returns:
        ``(成功输出路径列表, 失败信息列表, 汇总信息)``
    """
    images = list_image_files(input_dir)
    if not images:
        raise ImageSRError(f"目录中没有可处理的图片：{input_dir}")

    output_dir = config.OUTPUT_DIR / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    failures: list[str] = []
    started = time.time()
    total = len(images)

    for index, image_path in enumerate(images, start=1):
        label = f"[{index}/{total}] {image_path.name}"
        if progress:
            progress((index - 1) / total, f"处理 {label}")
        try:
            target, _info = upscale_path(
                image_path,
                output_dir=output_dir,
                image_format=image_format,
                progress=None,
                **kwargs,
            )
            outputs.append(target)
        except Exception as exc:  # noqa: BLE001
            logger.warning("批量处理失败：%s -> %s", image_path.name, exc)
            failures.append(f"{image_path.name}: {exc}")

    if progress:
        progress(1.0, f"批量完成：成功 {len(outputs)} / 失败 {len(failures)}")

    summary = {
        "total": total,
        "succeeded": len(outputs),
        "failed": len(failures),
        "elapsed": time.time() - started,
        "output_dir": str(output_dir),
    }
    return outputs, failures, summary


__all__ = [
    "ImageSRError",
    "ModelNotFoundError",
    "RealESRGANEngine",
    "FaceRestorer",
    "HAS_REALESRGAN",
    "HAS_CV2",
    "HAS_TORCH",
    "get_engine",
    "upscale_image",
    "upscale_path",
    "upscale_folder",
    "read_image",
    "write_image",
    "resize_to_long_edge",
    "resolve_target_long_edge",
    "available_base_models",
    "download_model",
    "ensure_base_model",
    "estimate_required_vram_gb",
    "build_network",
    "load_base_weights",
    "model_scale",
    "model_num_block",
]
