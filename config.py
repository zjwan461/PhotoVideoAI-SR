"""全局配置、路径、预设、GPU 探测。

本模块只做“纯配置 + 轻量探测”，不导入 torch / gradio 等重依赖，
因此可以被 UI、命令行、测试脚本安全复用。

设计约束：
* 代码中不出现任何特定显卡型号，GPU 信息一律运行时动态探测。
* 所有目录在导入时自动创建（已存在则忽略）。
* 所有可调项都支持环境变量覆盖，便于部署与自动化测试。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent

TEMP_DIR: Path = Path(os.environ.get("PVSR_TEMP_DIR", PROJECT_ROOT / "temp"))
OUTPUT_DIR: Path = Path(os.environ.get("PVSR_OUTPUT_DIR", PROJECT_ROOT / "output"))
LORA_DIR: Path = Path(os.environ.get("PVSR_LORA_DIR", PROJECT_ROOT / "lora_weights"))
MODEL_DIR: Path = Path(os.environ.get("PVSR_MODEL_DIR", PROJECT_ROOT / "model_weights"))
LOG_DIR: Path = Path(os.environ.get("PVSR_LOG_DIR", PROJECT_ROOT / "logs"))

#: 需要在启动时确保存在的目录
REQUIRED_DIRS = (TEMP_DIR, OUTPUT_DIR, LORA_DIR, MODEL_DIR, LOG_DIR)


def ensure_dirs() -> None:
    """创建所有必需目录（幂等），IO 失败不致命，只提示。"""
    for directory in REQUIRED_DIRS:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # 只读挂载、权限不足等
            print(f"[config] 无法创建目录 {directory}: {exc}", file=sys.stderr)


ensure_dirs()

# --------------------------------------------------------------------------- #
# FFmpeg / FFprobe
# --------------------------------------------------------------------------- #
#: 项目内自带的工具目录（运行时下载，不进版本库）
TOOLS_DIR: Path = Path(os.environ.get("PVSR_TOOLS_DIR", PROJECT_ROOT / "tools"))

#: 自带 FFmpeg 的安装目录
FFMPEG_BUNDLE_DIR: Path = TOOLS_DIR / "ffmpeg"


def bundled_ffmpeg_bin(name: str) -> Path:
    """返回项目内自带 FFmpeg 可执行文件路径（未下载时该文件不存在）。"""
    return FFMPEG_BUNDLE_DIR / (f"{name}.exe" if os.name == "nt" else name)


def _resolve_ffmpeg_bin(env_var: str, name: str) -> str:
    """决定用哪个 ffmpeg/ffprobe：环境变量 > 项目内自带 > PATH。"""
    override = os.environ.get(env_var)
    if override:
        return override
    bundled = bundled_ffmpeg_bin(name)
    if bundled.is_file():
        return str(bundled)
    return name


#: 可执行文件路径：默认走 PATH；装了项目内自带版本后自动指向它；
#: 也可用环境变量指定绝对路径（优先级最高）
FFMPEG_BIN: str = _resolve_ffmpeg_bin("PVSR_FFMPEG", "ffmpeg")
FFPROBE_BIN: str = _resolve_ffmpeg_bin("PVSR_FFPROBE", "ffprobe")


def refresh_ffmpeg_bins() -> tuple[str, str]:
    """重新解析 ffmpeg/ffprobe 路径。

    下载或删除项目内自带版本后调用，让 ``FFMPEG_BIN`` / ``FFPROBE_BIN`` 立刻生效
    （``src.ffmpeg_utils`` 每次调用都会读取这两个模块级变量，因此无需重启）。
    """
    global FFMPEG_BIN, FFPROBE_BIN
    FFMPEG_BIN = _resolve_ffmpeg_bin("PVSR_FFMPEG", "ffmpeg")
    FFPROBE_BIN = _resolve_ffmpeg_bin("PVSR_FFPROBE", "ffprobe")
    return FFMPEG_BIN, FFPROBE_BIN


#: 单条 ffmpeg 命令的超时（秒），防止损坏文件导致命令挂死
FFMPEG_TIMEOUT: int = int(os.environ.get("PVSR_FFMPEG_TIMEOUT", "3600"))

# --------------------------------------------------------------------------- #
# 分辨率预设
# --------------------------------------------------------------------------- #
#: 名称 -> (宽, 高)。等比缩放到该长边预算，保持原始宽高比。
RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "原始比例 (不限制长边)": (0, 0),
    "720p (1280x720)": (1280, 720),
    "1080p (1920x1080)": (1920, 1080),
    "2K (2560x1440)": (2560, 1440),
    "4K (3840x2160)": (3840, 2160),
    "8K (7680x4320)": (7680, 4320),
}

DEFAULT_RESOLUTION_PRESET: str = "原始比例 (不限制长边)"

# --------------------------------------------------------------------------- #
# 输出格式
# --------------------------------------------------------------------------- #
IMAGE_FORMATS: tuple[str, ...] = ("PNG", "JPG")
DEFAULT_IMAGE_FORMAT: str = "PNG"

VIDEO_FORMATS: tuple[str, ...] = ("MP4", "MKV")
DEFAULT_VIDEO_FORMAT: str = "MP4"

#: 容器 -> (视频编码容器参数, 音频编码)
VIDEO_CODEC_MAP: dict[str, dict[str, str]] = {
    "MP4": {"audio_codec": "aac", "audio_ext": "m4a"},
    "MKV": {"audio_codec": "copy", "audio_ext": "mka"},
}

# --------------------------------------------------------------------------- #
# 基础模型
# --------------------------------------------------------------------------- #
#: RealESRGAN 权重文件名 -> 官方下载地址（P1 自动下载使用）
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "RealESRGAN_x4plus.pth": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "desc": "通用 4 倍放大基础模型（默认）",
    },
    "RealESRGAN_x4plus_anime_6B.pth": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "desc": "动漫插画向 4 倍放大模型",
    },
    "RealESRGAN_x2plus.pth": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "desc": "通用 2 倍放大模型",
    },
}

DEFAULT_MODEL_NAME: str = "RealESRGAN_x4plus.pth"

#: 每个基础模型自身固有的放大倍数，用于计算 Lanczos 兜底缩放
MODEL_SCALE: dict[str, int] = {
    "RealESRGAN_x4plus.pth": 4,
    "RealESRGAN_x4plus_anime_6B.pth": 4,
    "RealESRGAN_x2plus.pth": 2,
}

#: 第三方人脸修复可选依赖（P1，可开关，缺库时自动隐藏）
GFPGAN_MODEL_URL: str = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
)

# --------------------------------------------------------------------------- #
# 老照片前置复原（P4-2，可选）
# --------------------------------------------------------------------------- #
#: 复原模型的权重目录（与「放大模型」分开，避免混在一起）
RESTORE_DIR: Path = Path(os.environ.get("PVSR_RESTORE_DIR", MODEL_DIR / "restore"))

#: 前置复原模型清单。复原一律**同尺寸**：只做去压缩块 / 去噪，放大仍交给超分模型。
#: 架构参数与官方 SwinIR 推理脚本一致，网络实现直接复用 basicsr.archs.swinir_arch。
RESTORE_MODELS: dict[str, dict] = {
    "jpeg": {
        "file": "swinir_colorcar_jpeg40.pth",
        "label": "去压缩块 · 彩色（JPEG / 扫图 / 网图，推荐）",
        "desc": "SwinIR-M 彩色 JPEG 压缩伪影去除（质量 40 档训练）",
        "size_mb": 98,
        "channels": "rgb",
        "url": (
            "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
            "006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth"
        ),
        "arch": {
            # 注意：CAR（去压缩块）系列的 img_range 是 255，而 DN（去噪）是 1，
            # 这是官方脚本里的说明（"use img_range=255 because it's slightly better than 1"）。
            # 取错会让输出整体偏移、画面严重失真。
            "upscale": 1, "in_chans": 3, "img_size": 126, "window_size": 7,
            "img_range": 255.0, "depths": [6] * 6, "embed_dim": 180,
            "num_heads": [6] * 6, "mlp_ratio": 2, "upsampler": "",
            "resi_connection": "1conv",
        },
        "scale": 1,
    },
    "gjpeg": {
        "file": "swinir_car_jpeg40.pth",
        "label": "去压缩块 · 黑白（只在亮度通道复原）",
        "desc": "SwinIR-M 灰度 JPEG 伪影去除；黑白老照片用它最合适，彩色图也能用",
        "size_mb": 98,
        "channels": "y",
        "url": (
            "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
            "006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth"
        ),
        "arch": {
            "upscale": 1, "in_chans": 1, "img_size": 126, "window_size": 7,
            "img_range": 255.0, "depths": [6] * 6, "embed_dim": 180,
            "num_heads": [6] * 6, "mlp_ratio": 2, "upsampler": "",
            "resi_connection": "1conv",
        },
        "scale": 1,
    },
    "denoise": {
        "file": "swinir_colordn_noise25.pth",
        "label": "去噪 · 彩色（胶片颗粒 / 高感噪点）",
        "desc": "SwinIR-M 彩色图像去噪（噪声等级 25）",
        "size_mb": 117,
        "channels": "rgb",
        "url": (
            "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
            "005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth"
        ),
        "arch": {
            "upscale": 1, "in_chans": 3, "img_size": 128, "window_size": 8,
            "img_range": 1.0, "depths": [6] * 6, "embed_dim": 180,
            "num_heads": [6] * 6, "mlp_ratio": 2, "upsampler": "",
            "resi_connection": "1conv",
        },
        "scale": 1,
    },
}

#: 分块推理尺寸（SwinIR 显存随边长平方增长，大图必须分块）
RESTORE_TILE: int = int(os.environ.get("PVSR_RESTORE_TILE", "384"))

#: 不在下拉里展示复原功能
NO_RESTORE: str = "不使用前置复原"


def restore_model_path(key: str) -> Path:
    """某个复原模型的权重路径。"""
    entry = RESTORE_MODELS.get(key)
    name = entry["file"] if entry else key
    return RESTORE_DIR / name


def restore_installed(key: str) -> bool:
    """权重是否已下载。"""
    return key in RESTORE_MODELS and restore_model_path(key).is_file()


def restore_choices() -> list[str]:
    """生成 UI 下拉选项：占位项 + 已安装的复原模型。"""
    options = [NO_RESTORE]
    for key, entry in RESTORE_MODELS.items():
        if restore_installed(key):
            options.append(f"{entry['label']} [{key}]")
    return options


def parse_restore_choice(choice: str | None) -> str | None:
    """把下拉选项解析成模型 key；占位项或非法输入返回 None。"""
    if not choice or choice == NO_RESTORE:
        return None
    for key, entry in RESTORE_MODELS.items():
        if choice == f"{entry['label']} [{key}]":
            return key
    # 容错：允许直接传 key
    return choice if choice in RESTORE_MODELS else None


def restore_available() -> bool:
    """是否至少装了一个复原模型（用于决定 UI 是否显示该控件）。"""
    return any(restore_installed(key) for key in RESTORE_MODELS)

# --------------------------------------------------------------------------- #
# 全局运行参数
# --------------------------------------------------------------------------- #
HALF_PRECISION: bool = os.environ.get("PVSR_HALF", "1") not in ("0", "false", "False")
KEEP_TEMP_FILES: bool = os.environ.get("PVSR_KEEP_TEMP", "0") in ("1", "true", "True")

#: FFmpeg 软件编码预设（硬件编码走 nvenc 时仅作回退用）
FFMPEG_X264_PRESET: str = os.environ.get("PVSR_X264_PRESET", "medium")
FFMPEG_CRF: int = int(os.environ.get("PVSR_CRF", "17"))

#: 输出码率（Mbps）；为 None 时使用 CRF 质量控制
DEFAULT_VIDEO_BITRATE_MBPS: float | None = None

#: 视频默认帧率（当源视频探测不到 fps 时使用）
DEFAULT_FPS: float = 25.0


def video_pipeline() -> str:
    """返回视频处理管线：``stream``（默认，管道式）或 ``files``（拆帧成 PNG）。

    设环境变量 ``PVSR_VIDEO_PIPELINE=files`` 可强制使用文件管线，便于对比或排查。
    """
    value = os.environ.get("PVSR_VIDEO_PIPELINE", "stream").strip().lower()
    if value in {"files", "legacy", "0", "false", "off"}:
        return "files"
    return "stream"

#: 单张图片/单帧推理的分块大小，越小越省显存
TILE_SIZE: int = int(os.environ.get("PVSR_TILE", "0"))  # 0 = 自动
TILE_PAD: int = int(os.environ.get("PVSR_TILE_PAD", "10"))

#: 预留安全显存余量（GiB），OOM 预判时扣除
VRAM_SAFETY_MARGIN_GB: float = float(os.environ.get("PVSR_VRAM_MARGIN", "1.0"))

# --------------------------------------------------------------------------- #
# P1：预览模式
# --------------------------------------------------------------------------- #
#: 预览模式下最多渲染的帧数
PREVIEW_FRAME_LIMIT: int = int(os.environ.get("PVSR_PREVIEW_FRAMES", "30"))

# --------------------------------------------------------------------------- #
# GPU 探测
# --------------------------------------------------------------------------- #
#: torch 不可用时的设备列表，保证 UI 仍能渲染（推理时会给出明确报错）
NO_TORCH_DEVICES: list[str] = ["CPU (未检测到可用 CUDA)"]


def torch_available() -> bool:
    """返回 torch 是否可导入（不触发 CUDA 初始化报错）。"""
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def cuda_available() -> bool:
    """返回 torch 是否能真正使用 CUDA。"""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def detect_cuda_devices() -> list[dict]:
    """动态探测所有可见 CUDA 设备。

    返回列表，每项包含 index / name / total_vram_gb / free_vram_gb / capability。
    没有 torch 或没有 CUDA 时返回空列表。
    """
    devices: list[dict] = []
    try:
        import torch
    except Exception:
        return devices

    try:
        if not torch.cuda.is_available():
            return devices
        count = torch.cuda.device_count()
        for index in range(count):
            try:
                props = torch.cuda.get_device_properties(index)
                name = getattr(props, "name", f"CUDA:{index}")
                capability = torch.cuda.get_device_capability(index)
                total_gb = round(props.total_memory / (1024**3), 1)
                try:
                    free_bytes, _total = torch.cuda.mem_get_info(index)
                    free_gb = round(free_bytes / (1024**3), 1)
                except Exception:
                    free_gb = total_gb
                devices.append(
                    {
                        "index": index,
                        "name": name,
                        "total_vram_gb": total_gb,
                        "free_vram_gb": free_gb,
                        "capability": f"sm_{capability[0]}{capability[1]}",
                    }
                )
            except Exception:
                # 单卡探测失败不影响其他卡
                continue
    except Exception:
        return []
    return devices


def get_gpu_options() -> list[str]:
    """生成 Gradio 下拉框用的设备选项列表。

    形如 ``["cuda:0 | <运行时探测到的名称> (24.0 GB)", "cuda:1 | ..."]``。
    无可用 CUDA 时返回提示项，主程序会在启动阶段直接退出并给出安装指引。
    """
    devices = detect_cuda_devices()
    if not devices:
        return list(NO_TORCH_DEVICES)
    return [
        f"cuda:{d['index']} | {d['name']} ({d['total_vram_gb']} GB, {d['capability']})"
        for d in devices
    ]


def parse_gpu_option(option: str | None) -> str:
    """把下拉框字符串解析回 ``cuda:N``；非法输入回退到 ``cuda:0``。"""
    if not option:
        return "cuda:0"
    token = str(option).strip().split("|")[0].strip()
    if token.startswith("cuda:"):
        try:
            int(token.split(":", 1)[1])
        except ValueError:
            return "cuda:0"
        return token
    return "cuda:0"


def gpu_index_from_option(option: str | None) -> int:
    """从下拉框字符串解析出设备序号。"""
    return int(parse_gpu_option(option).split(":", 1)[1])


def normalize_device(device: str | None) -> str:
    """把 UI 下拉标签 / 简写统一成 ``torch.device()`` 能接受的设备字符串。

    项目里 ``device`` 有两种来源：

    * Gradio 下拉框的**显示标签**，形如
      ``"cuda:0 | NVIDIA GeForce RTX 4080 SUPER (32.0 GB, sm_89)"``；
    * 代码里的简写，如 ``"cuda:0"``、``"cuda"``、``"cpu"``。

    两者都允许传入，但**交给 torch 之前必须归一化**：把标签原样传下去会得到
    ``RuntimeError: Invalid device string: 'cuda:0 | ...'``。

    Args:
        device: 下拉标签、``cuda:N``、``cuda``、``cpu``，或 None/空值。

    Returns:
        归一化后的设备字符串。``cpu`` 会被保留（供 ``--allow-cpu`` 调试与
        CPU 自测使用）；空值按 CUDA 可用性自动选择。
    """
    if not device:
        return "cuda:0" if cuda_available() else "cpu"
    token = str(device).strip().split("|")[0].strip().lower()
    if token.startswith("cuda"):
        if ":" not in token:
            return "cuda:0"
        _, _, index = token.partition(":")
        try:
            int(index)
        except ValueError:
            return "cuda:0"
        return f"cuda:{index}"
    if token == "cpu":
        return "cpu"
    return "cuda:0" if cuda_available() else "cpu"


# --------------------------------------------------------------------------- #
# 外部依赖可用性
# --------------------------------------------------------------------------- #
def ffmpeg_available() -> bool:
    """检测 ffmpeg / ffprobe 是否可执行（PATH 或显式路径均可）。"""
    return bool(shutil.which(FFMPEG_BIN)) and bool(shutil.which(FFPROBE_BIN))


def ffmpeg_diagnosis() -> str:
    """返回面向 UI 的 FFmpeg 状态文本。"""
    missing = []
    if not shutil.which(FFMPEG_BIN):
        missing.append(FFMPEG_BIN)
    if not shutil.which(FFPROBE_BIN):
        missing.append(FFPROBE_BIN)
    if not missing:
        return "✅ FFmpeg / FFprobe 已就绪"
    return (
        "❌ 未找到可执行文件: "
        + ", ".join(missing)
        + "\n可在「🩺 环境与 LoRA」Tab 点「⬇️ 下载 FFmpeg」，程序会自动下载到项目内"
        f"\n（{FFMPEG_BUNDLE_DIR}）并立即使用，无需管理员权限、也不用改系统 PATH。"
        "\n也可以自行安装 FFmpeg 并加入 PATH，或用环境变量 PVSR_FFMPEG / PVSR_FFPROBE 指定绝对路径。"
    )


def maxine_available() -> bool:
    """检测 NVIDIA Maxine VSR（nvvfx）是否可用。"""
    try:
        import nvvfx  # noqa: F401
    except Exception:
        return False
    return True


def gfpgan_weights_path() -> Path:
    """人脸修复（可选组件）的权重路径。"""
    return MODEL_DIR / "GFPGANv1.4.pth"


#: GFPGAN 通过 facexlib 加载人脸检测 / 解析模型的目录。
#: 用**项目内绝对路径**而不是 facexlib 包目录：``gfpgan/utils.py`` 传给 facexlib 的
#: ``model_rootpath='gfpgan/weights'`` 是相对路径，facexlib 会按当前工作目录解析，
#: 从别的目录启动就会重复下载。钉在项目内才与启动目录无关（见 src/gfpgan_compat.py）。
GFPGAN_WEIGHTS_DIR: Path = PROJECT_ROOT / "gfpgan" / "weights"


#: GFPGAN 还会通过 facexlib 加载人脸**检测 / 解析**模型。
#: facexlib 在首次使用时才去 GitHub 下载（约 104 MB + 81 MB），
#: 国内网络下会长时间卡住且看不出原因，因此由安装脚本预先下载。
GFPGAN_FACEXLIB_MODELS: dict[str, str] = {
    "detection_Resnet50_Final.pth": (
        "https://github.com/xinntao/facexlib/releases/download/v0.1.0/"
        "detection_Resnet50_Final.pth"
    ),
    "parsing_parsenet.pth": (
        "https://github.com/xinntao/facexlib/releases/download/v0.2.2/"
        "parsing_parsenet.pth"
    ),
}


def facexlib_weights_dir() -> Path:
    """GFPGAN / facexlib 人脸检测与解析模型的权重目录（绝对路径）。"""
    return GFPGAN_WEIGHTS_DIR


def missing_facexlib_models() -> list[str]:
    """返回还缺哪些 facexlib 模型（空列表表示齐全）。"""
    directory = facexlib_weights_dir()
    return [name for name in GFPGAN_FACEXLIB_MODELS if not (directory / name).is_file()]


def gfpgan_available() -> bool:
    """检测可选的人脸修复是否**真正可用**：库、GFPGAN 权重、facexlib 模型都要在。

    任何一项缺失都返回 False，这样 UI 会隐藏该开关，而不是让人点下去才报错
    （facexlib 缺模型时甚至会卡在联网下载上，表现成「没反应」）。

    注意：``gfpgan`` 会连带导入 ``basicsr``，而后者仍在导入 torchvision 已删除的
    ``transforms.functional_tensor``。这里先装兼容垫片再导入，否则即使库和权重都在，
    这个开关也永远不出现。
    """
    try:
        try:
            from src.torchvision_compat import ensure_functional_tensor

            ensure_functional_tensor()
        except Exception:  # noqa: BLE001 - 垫片不可用不应影响可用性判断
            pass
        from gfpgan import GFPGANer  # noqa: F401
    except Exception:
        return False
    if not gfpgan_weights_path().is_file():
        return False
    return not missing_facexlib_models()


def video_model_options() -> list[str]:
    """动态生成视频模型下拉选项：RealESRGAN 始终存在，Maxine 可用时才追加。"""
    options = ["RealESRGAN (逐帧超分, 可微调)"]
    if maxine_available():
        options.append("Maxine VSR (时序超分, 需要 nvvfx)")
    return options


def runtime_summary() -> str:
    """生成一段运行环境摘要，用于日志与 UI 信息面板。"""
    lines = [f"Python: {sys.version.split()[0]}", f"FFmpeg: {FFMPEG_BIN}", ffmpeg_diagnosis()]
    devices = detect_cuda_devices()
    if devices:
        for device in devices:
            lines.append(
                f"CUDA:{device['index']} {device['name']} "
                f"{device['total_vram_gb']} GB ({device['capability']})"
            )
    else:
        lines.append("CUDA: 未检测到可用设备")
    lines.append(f"Maxine VSR: {'可用' if maxine_available() else '不可用 (已隐藏)'}")
    installed_restore = [key for key in RESTORE_MODELS if restore_installed(key)]
    if installed_restore:
        lines.append(f"前置复原: 可用 ({'、'.join(installed_restore)})")
    else:
        lines.append("前置复原: 不可用 (未下载权重，已隐藏)")
    if gfpgan_available():
        lines.append("GFPGAN: 可用")
    elif not gfpgan_weights_path().is_file():
        lines.append(f"GFPGAN: 不可用 (缺少权重 {gfpgan_weights_path().name}，已隐藏)")
    elif missing := missing_facexlib_models():
        lines.append(f"GFPGAN: 不可用 (缺少 facexlib 模型 {'、'.join(missing)}，已隐藏)")
    else:
        lines.append("GFPGAN: 不可用 (缺少 gfpgan 依赖，已隐藏)")
    return "\n".join(lines)
