"""通用工具：CUDA/显存检测、显存预判、目录清理、LoRA 扫描、日志。

本模块可被 UI 与推理模块共用，除 torch 外不引入重依赖（torch 亦为惰性导入）。
"""

from __future__ import annotations

import logging
import os
import shutil
import time
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import config

# --------------------------------------------------------------------------- #
# 日志
# --------------------------------------------------------------------------- #
_LOGGER_NAME = "photovideoai_sr"
_LOGGER: logging.Logger | None = None

#: 任务级日志文件（每次进程启动写一个新的带时间戳的文件）
TASK_LOG_FILE: Path = config.LOG_DIR / f"session_{datetime.now():%Y%m%d_%H%M%S}.log"


def get_logger(name: str = "app") -> logging.Logger:
    """返回全局 logger（幂等初始化，同时输出到控制台与 logs/ 目录）。"""
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    try:
        file_handler = RotatingFileHandler(
            TASK_LOG_FILE, maxBytes=4 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        print(f"[common] 无法写入日志文件 {TASK_LOG_FILE}: {exc}")

    _LOGGER = logger
    return logger


def log_runtime_environment() -> None:
    """把运行环境（Python / FFmpeg / GPU / 可选组件）写入日志头部。"""
    logger = get_logger("env")
    logger.info("=" * 68)
    logger.info("PhotoVideoAI-SR 启动，运行环境摘要：")
    for line in config.runtime_summary().splitlines():
        logger.info("  %s", line)
    logger.info("=" * 68)


def log_task(task: str, params: dict, elapsed: float | None = None, gpu: str | None = None) -> None:
    """记录一次任务的参数、耗时与设备信息。"""
    logger = get_logger("task")
    parts = [f"任务={task}"]
    if gpu:
        parts.append(f"设备={gpu}")
    parts.append("参数=" + ", ".join(f"{k}={v!r}" for k, v in params.items()))
    if elapsed is not None:
        parts.append(f"耗时={elapsed:.2f}s")
    logger.info(" | ".join(parts))


def log_exception(context: str, exc: BaseException) -> None:
    """记录异常与堆栈。"""
    get_logger("error").error("%s 失败: %s\n%s", context, exc, traceback.format_exc())


# --------------------------------------------------------------------------- #
# CUDA / 显存
# --------------------------------------------------------------------------- #
def cuda_available() -> bool:
    """torch 是否能真正使用 CUDA。"""
    return config.cuda_available()


def require_cuda(exit_on_failure: bool = False) -> bool:
    """检查 CUDA；缺失时打印可读提示并可选直接退出进程。"""
    if cuda_available():
        return True
    message = (
        "未检测到可用的 CUDA 设备。\n"
        "本项目仅支持 NVIDIA CUDA 推理，请在具备 CUDA 环境的机器上运行：\n"
        "  1. 安装与驱动匹配的 NVIDIA 驱动；\n"
        "  2. 安装 CUDA 版 PyTorch（见 README「安装」章节）；\n"
        "  3. 运行 `python -c \"import torch; print(torch.cuda.is_available())\"` 自检。"
    )
    print(message)
    if exit_on_failure:
        raise SystemExit(1)
    return False


def get_device_summary(device: str = "cuda:0") -> str:
    """返回单行设备描述，用于日志/UI（避免硬编码任何型号）。

    接受 UI 下拉标签（``"cuda:0 | <名称> (32 GB, sm_89)"``）或 ``cuda:N`` /
    ``cpu``；内部先归一化再拼装，避免把标签原样拼进日志。
    """
    normalized = config.normalize_device(device)
    try:
        import torch

        if normalized.startswith("cuda") and torch.cuda.is_available():
            index = config.gpu_index_from_option(normalized)
            props = torch.cuda.get_device_properties(index)
            return (
                f"{normalized} | {getattr(props, 'name', 'CUDA')} "
                f"({round(props.total_memory / 1024**3, 1)} GB)"
            )
    except Exception:
        pass
    return normalized


def get_free_vram_gb(device_index: int = 0) -> float | None:
    """获取目标 GPU 的空闲显存（GiB）；不可用时返回 None。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_bytes, _total = torch.cuda.mem_get_info(device_index)
        return round(free_bytes / (1024**3), 2)
    except Exception:
        return None


def get_total_vram_gb(device_index: int = 0) -> float | None:
    """获取目标 GPU 的总显存（GiB）；不可用时返回 None。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(device_index)
        return round(props.total_memory / (1024**3), 2)
    except Exception:
        return None


def check_vram_for_task(
    required_gb: float,
    device_option: str | None = None,
    task_label: str = "当前任务",
) -> tuple[bool, str]:
    """推理/训练前的显存预判。

    Args:
        required_gb: 预估所需显存（GiB）。
        device_option: Gradio 下拉框里的设备字符串。
        task_label: 用于提示文案的任务名。

    Returns:
        ``(ok, message)``；``ok=False`` 时 message 为面向用户的警告文案。
    """
    if not cuda_available():
        return False, "未检测到可用 CUDA 设备，无法执行 GPU 任务。"
    index = config.gpu_index_from_option(device_option)
    free_gb = get_free_vram_gb(index)
    total_gb = get_total_vram_gb(index)
    if free_gb is None:
        return True, "无法读取显存信息，已跳过显存预判。"

    usable_gb = free_gb - config.VRAM_SAFETY_MARGIN_GB
    if usable_gb < required_gb:
        return (
            False,
            (
                f"⚠️ 显存可能不足：{task_label} 预估需要约 {required_gb:.1f} GB，"
                f"当前 cuda:{index} 空闲 {free_gb:.1f} GB / 总计 {total_gb if total_gb else '未知'} GB"
                f"（已预留 {config.VRAM_SAFETY_MARGIN_GB:.1f} GB 安全余量）。\n"
                "建议：关闭其它占用显存的程序、调小 tile 尺寸、改用更小的模型或开启预览模式。"
            ),
        )
    return True, f"显存检查通过：cuda:{index} 空闲 {free_gb:.1f} GB，预估需要 {required_gb:.1f} GB。"


def estimate_image_vram_gb(scale_hint: int = 1) -> float:
    """粗略估算单张图片超分所需显存（非常保守，仅用于预判提示）。"""
    return round(2.0 + 1.0 * max(1, int(scale_hint)), 1)


def friendly_cuda_error(exc: BaseException) -> str | None:
    """把 CUDA OOM 等异常转成可读提示；非显存异常返回 None。"""
    text = str(exc).lower()
    if "out of memory" in text or "cuda oom" in text or "cublas" in text and "alloc" in text:
        return (
            "❌ GPU 显存不足（CUDA Out of Memory）。\n"
            "建议：降低输出分辨率 / 调小 tile 尺寸 / 关闭 half 精度 / 先关闭其它占用显存的程序，然后重试。"
        )
    if "no kernel image is available" in text or "not compiled with cuda" in text:
        return (
            "❌ 当前 PyTorch 与这台机器的 CUDA 环境不匹配。\n"
            "请安装 CUDA 版 PyTorch，并确认驱动版本满足该 PyTorch 版本的要求。"
        )
    if "cuda driver version is insufficient" in text or "no cuda-capable device" in text:
        return "❌ CUDA 驱动不可用或版本过低，请更新 NVIDIA 驱动后重试。"
    return None


def clear_cuda_cache() -> None:
    """清理 torch 的显存缓存（任务结束或 OOM 之后调用）。"""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# 临时文件
# --------------------------------------------------------------------------- #
def make_temp_subdir(prefix: str = "task") -> Path:
    """在 temp/ 下创建一个唯一子目录，返回其路径。"""
    stamp = f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}_{int(time.time() * 1000) % 1000:03d}"
    path = config.TEMP_DIR / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_remove(path: str | Path) -> bool:
    """安全删除文件或目录（不跟随危险路径），返回是否删除了内容。"""
    try:
        target = Path(path).resolve()
    except OSError:
        return False

    # 只允许删除 temp / output 内的内容，避免误删用户目录
    allowed_roots = (config.TEMP_DIR.resolve(), config.OUTPUT_DIR.resolve())
    if not any(str(target).startswith(str(root)) for root in allowed_roots):
        return False
    if target in (config.TEMP_DIR.resolve(), config.OUTPUT_DIR.resolve()):
        return False

    try:
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()
        else:
            return False
        return True
    except OSError:
        return False


def cleanup_temp_dir(keep: bool | None = None) -> tuple[int, int]:
    """清理 temp 目录（任务成功或失败后都应调用）。

    Args:
        keep: 为 True 时保留（调试用）；None 时读取 config.KEEP_TEMP_FILES。

    Returns:
        ``(removed_entries, failed_entries)``
    """
    keep = config.KEEP_TEMP_FILES if keep is None else keep
    if keep:
        get_logger("cleanup").info("保留临时文件（已开启保留开关）：%s", config.TEMP_DIR)
        return 0, 0

    removed = failed = 0
    if not config.TEMP_DIR.exists():
        return 0, 0
    for entry in config.TEMP_DIR.iterdir():
        if entry.name == ".gitkeep":
            continue
        try:
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=False)
            else:
                entry.unlink()
            removed += 1
        except OSError:
            failed += 1
    get_logger("cleanup").info("临时目录清理完成：删除 %d 项，失败 %d 项", removed, failed)
    return removed, failed


def cleanup_stale_temp(max_age_hours: float = 24.0) -> int:
    """清理 temp 目录中超过 max_age_hours 的残留（防止崩溃后无限堆积）。"""
    if not config.TEMP_DIR.exists():
        return 0
    deadline = time.time() - max_age_hours * 3600
    removed = 0
    for entry in config.TEMP_DIR.iterdir():
        if entry.name == ".gitkeep":
            continue
        try:
            if entry.stat().st_mtime < deadline:
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink()
                removed += 1
        except OSError:
            continue
    if removed:
        get_logger("cleanup").info("清理历史残留临时项 %d 个", removed)
    return removed


# --------------------------------------------------------------------------- #
# LoRA
# --------------------------------------------------------------------------- #
def scan_lora_weights(lora_dir: Path | None = None) -> list[str]:
    """扫描 lora_weights/ 目录，返回可用 LoRA 文件名列表（供下拉框）。

    只返回 *.pth / *.pt / *.safetensors；结果按名称排序。
    """
    directory = Path(lora_dir) if lora_dir else config.LORA_DIR
    if not directory.exists():
        return []
    suffixes = {".pth", ".pt", ".safetensors"}
    names = sorted(
        entry.name
        for entry in directory.iterdir()
        if entry.is_file()
        and entry.suffix.lower() in suffixes
        and not entry.name.startswith(".")
        # 续训状态文件（*.state.pt）不是可部署的 LoRA，不进入下拉框
        and not entry.name.endswith(".state.pt")
    )
    return names


def lora_choices(include_none: bool = True) -> list[str]:
    """Gradio 下拉框选项：``["不使用 LoRA", ...文件名]``。"""
    items = scan_lora_weights()
    if include_none:
        return ["不使用 LoRA"] + items
    return items


def resolve_lora_path(selection: str | None) -> Path | None:
    """把下拉框选项解析为绝对路径；选择「不使用」或文件不存在时返回 None。"""
    if not selection or selection.startswith("不使用"):
        return None
    candidate = Path(selection)
    if not candidate.is_absolute():
        candidate = config.LORA_DIR / selection
    try:
        candidate = candidate.resolve()
    except OSError:
        return None
    # 限定在 lora_weights/ 内，避免任意路径读取
    try:
        candidate.relative_to(config.LORA_DIR.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def config_lora_choices() -> list[str]:
    """扫描 model_weights/ 下作为「配置 LoRA」用途的权重文件。"""
    suffixes = {".pth", ".pt", ".safetensors"}
    if not config.MODEL_DIR.exists():
        return ["(无)"]
    items = sorted(
        entry.name
        for entry in config.MODEL_DIR.iterdir()
        if entry.is_file() and entry.suffix.lower() in suffixes
    )
    return ["(无)"] + items


# --------------------------------------------------------------------------- #
# 其它小工具
# --------------------------------------------------------------------------- #
def unique_output_path(stem: str, suffix: str, subdir: str | None = None) -> Path:
    """在 output/ 下生成不冲突的输出路径。"""
    directory = config.OUTPUT_DIR / subdir if subdir else config.OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem) or "output"
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    candidate = directory / f"{safe_stem}_{stamp}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{safe_stem}_{stamp}_{counter}{suffix}"
        counter += 1
    return candidate


def human_size(num_bytes: float) -> str:
    """字节数转可读字符串。"""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def elapsed_str(seconds: float) -> str:
    """秒数转 ``1h02m03s`` 形式。"""
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def list_image_files(directory: str | Path) -> list[Path]:
    """列出目录下的图片文件（用于批量模式）。"""
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    path = Path(directory)
    if not path.exists():
        return []
    return sorted(
        entry for entry in path.rglob("*") if entry.is_file() and entry.suffix.lower() in suffixes
    )
