"""老照片附加增强（可选开关，主链路之外的“锦上添花”功能）。

包含三件事，,**都遵守「缺失即隐藏/降级」原则**：

1. :func:`denoise_image` —— 轻量去噪，基于 OpenCV 非局部均值，无需额外权重或依赖；
2. :func:`reduce_scratch` —— 轻量去划痕/去尘点，基于中值 + 非局部均值组合，
   对细划痕与噪点有效，但**不是** AI 级的划痕修复，请勿夸大预期；
3. :func:`colorize_image` —— 黑白上色接口。上色必须依赖外部彩色化模型，
   本项目不内置权重，因此这里提供**可插拔注册表**：装好引擎后调用
   :func:`register_colorizer` 注册，UI 会自动出现上色开关；未注册时开关隐藏。

所有函数只接受/返回 RGB numpy 数组，不写盘、不改动输入对象。
"""

from __future__ import annotations

from collections.abc import Callable

from src.common import get_logger

logger = get_logger("extras")

try:
    import cv2

    HAS_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    HAS_CV2 = False

try:
    import numpy as np

    HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]
    HAS_NUMPY = False


class ExtrasError(RuntimeError):
    """附加增强功能相关的可读错误。"""


def _require_cv2():
    if not HAS_CV2 or not HAS_NUMPY:
        raise ExtrasError(
            "缺少图像处理依赖（opencv-python / numpy），无法使用附加增强功能。\n"
            "请执行：pip install -r requirements.txt"
        )
    return cv2


# --------------------------------------------------------------------------- #
# 去噪 / 去划痕
# --------------------------------------------------------------------------- #
def denoise_image(image_rgb, strength: float = 5.0, color_strength: float | None = None):
    """轻量去噪（非局部均值），适合老旧照片的颗粒与压缩噪点。

    Args:
        image_rgb: HxWx3 或 HxWx4 的 RGB uint8 数组。
        strength: 亮度通道滤波强度，1~15；越大越平滑，也越容易糊掉细节。
        color_strength: 色度通道强度，默认取 ``strength``。

    Returns:
        去噪后的 RGB(A) 数组（尺寸不变）。
    """
    _require_cv2()
    if image_rgb is None or getattr(image_rgb, "size", 0) == 0:
        raise ExtrasError("输入图片为空。")

    value = float(max(1.0, min(15.0, strength)))
    color_value = float(max(1.0, min(15.0, strength if color_strength is None else color_strength)))

    has_alpha = image_rgb.ndim == 3 and image_rgb.shape[2] == 4
    rgb = image_rgb[:, :, :3] if has_alpha else image_rgb
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # 非局部均值要求 8UC1/8UC3，先做一次安全转换
    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoisingColored(
        bgr, None, value, color_value, 7, 21
    )
    result = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)

    if has_alpha:
        result = np.dstack([result, image_rgb[:, :, 3]])
    return result


def reduce_scratch(image_rgb, median_size: int = 3, denoise_strength: float = 4.0):
    """轻量去划痕/去尘点：中值滤波压制细线状缺损 + 非局部均值收尾。

    说明：这是传统图像处理方案，对**浅色细划痕与尘点**有效；
    大面积破损/折痕请配合人脸修复与手动修补，效果有限。
    """
    _require_cv2()
    if image_rgb is None or getattr(image_rgb, "size", 0) == 0:
        raise ExtrasError("输入图片为空。")

    size = int(median_size)
    if size not in (3, 5, 7):
        size = 3

    has_alpha = image_rgb.ndim == 3 and image_rgb.shape[2] == 4
    rgb = image_rgb[:, :, :3] if has_alpha else image_rgb
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)

    cleaned = cv2.medianBlur(bgr, size)
    cleaned = cv2.fastNlMeansDenoisingColored(
        cleaned, None, float(denoise_strength), float(denoise_strength), 7, 21
    )
    result = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)

    if has_alpha:
        result = np.dstack([result, image_rgb[:, :, 3]])
    return result


def apply_extras(
    image_rgb,
    *,
    denoise: bool = False,
    denoise_strength: float = 5.0,
    scratch: bool = False,
):
    """按开关依次应用附加增强（供 UI 直接调用）。"""
    result = image_rgb
    if denoise:
        result = denoise_image(result, strength=denoise_strength)
    if scratch:
        result = reduce_scratch(result)
    return result


# --------------------------------------------------------------------------- #
# 上色（可插拔后端）
# --------------------------------------------------------------------------- #
#: 注册的上色后端：调用签名 ``fn(image_rgb) -> image_rgb``
_COLORIZER: Callable | None = None
_COLORIZER_NAME: str = ""


def register_colorizer(func: Callable, name: str = "custom") -> None:
    """注册上色后端。装好彩色化模型后调用它，UI 即可显示上色开关。"""
    global _COLORIZER, _COLORIZER_NAME
    if not callable(func):
        raise ExtrasError("上色后端必须是可调用对象，签名 fn(image_rgb) -> image_rgb。")
    _COLORIZER = func
    _COLORIZER_NAME = name
    logger.info("已注册上色后端：%s", name)


def colorizer_available() -> bool:
    """当前是否可用上色。"""
    return _COLORIZER is not None


def colorizer_name() -> str:
    """当前上色后端名称（未注册时为空串）。"""
    return _COLORIZER_NAME


def colorize_image(image_rgb):
    """调用已注册的后端给黑白照片上色。"""
    if _COLORIZER is None:
        raise ExtrasError(
            "未安装上色引擎，无法上色。\n"
            "上色需要外部彩色化模型（例如 DDColor / DeOldify 一类），本项目不内置其权重：\n"
            "  1. 按所选引擎的官方说明安装依赖与权重；\n"
            "  2. 在启动前调用 src.enhance_extras.register_colorizer(fn, 'DDColor') 注册；\n"
            "  3. 重新启动 WebUI，上色开关会出现。"
        )
    if image_rgb is None or getattr(image_rgb, "size", 0) == 0:
        raise ExtrasError("输入图片为空。")
    return _COLORIZER(image_rgb)


def extras_status() -> str:
    """返回附加增强功能的可用性说明（供 UI 展示）。"""
    lines = [
        f"去噪 / 轻量去划痕：{'✅ 可用' if HAS_CV2 and HAS_NUMPY else '❌ 缺少 opencv-python'}",
        f"黑白上色：{'✅ 可用 (' + _COLORIZER_NAME + ')' if colorizer_available() else '❌ 未注册上色后端（UI 已隐藏该开关）'}",
    ]
    return "\n".join(lines)


def try_autoregister_colorizer() -> bool:
    """尝试自动发现常见上色后端（可选依赖）。找到就注册，找不到静默跳过。"""
    try:  # 例：modelscope 的 DDColor 管线
        from modelscope.pipelines import pipeline  # type: ignore

        model_id = "damo/cv_ddcolor_image-colorization"
        pipe = pipeline("image-colorization", model=model_id)

        def _infer(image_rgb):
            output = pipe(image_rgb)
            for key in ("output_img", "output", "image"):
                if isinstance(output, dict) and key in output:
                    candidate = output[key]
                    return candidate if hasattr(candidate, "shape") else np.asarray(candidate)
            raise ExtrasError("上色后端返回结果无法解析。")

        register_colorizer(_infer, "DDColor (modelscope)")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("未发现可用的上色后端（正常现象）：%s", exc)
        return False


__all__ = [
    "HAS_CV2",
    "ExtrasError",
    "denoise_image",
    "reduce_scratch",
    "apply_extras",
    "register_colorizer",
    "colorize_image",
    "colorizer_available",
    "colorizer_name",
    "extras_status",
    "try_autoregister_colorizer",
]
