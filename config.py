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
#: 可执行文件路径，默认走 PATH；可用环境变量指定绝对路径
FFMPEG_BIN: str = os.environ.get("PVSR_FFMPEG", "ffmpeg")
FFPROBE_BIN: str = os.environ.get("PVSR_FFPROBE", "ffprobe")

#: 单条 ffmpeg 命令的超时（秒），防止损坏文件导致进程挂死
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
        + "\n请安装 FFmpeg 并加入 PATH，或用环境变量 PVSR_FFMPEG / PVSR_FFPROBE 指定绝对路径。"
    )


def maxine_available() -> bool:
    """检测 NVIDIA Maxine VSR（nvvfx）是否可用。"""
    try:
        import nvvfx  # noqa: F401
    except Exception:
        return False
    return True


def gfpgan_available() -> bool:
    """检测可选的人脸修复依赖是否可用。"""
    try:
        from gfpgan import GFPGANer  # noqa: F401
    except Exception:
        return False
    return True


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
    lines.append(f"GFPGAN: {'可用' if gfpgan_available() else '不可用 (已隐藏)'}")
    return "\n".join(lines)
