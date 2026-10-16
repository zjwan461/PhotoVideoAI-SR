"""NVIDIA Maxine Video Super Resolution（时序视频超分，**可选组件**）。

关于本模块的三个重要事实
------------------------
1. **闭源**：Maxine VSR 是 NVIDIA 提供的闭源 SDK（通过 ``nvvfx`` 包调用），
   模型内部参数不可见、不可训练。因此 **它不支持 LoRA 微调**，
   UI 中也不会为它显示 LoRA 下拉框 —— 需要模型定制时请使用 RealESRGAN 链路。
2. **可选**：未安装 ``nvvfx`` 时，本模块导入不报错，只是把
   :data:`HAS_MAXINE` 置为 ``False``；UI 会自动隐藏 Maxine 选项，主流程不受影响。
3. **实时/低延迟导向**：逐帧按顺序送入即可获得时序增强效果，适合视频播放类场景；
   画质风格与 RealESRGAN 不同，二者按需选择（见 README「模型对比」）。

依赖安装（需要 NVIDIA 官方 SDK 与对应驱动，详见官方文档）：
    pip install nvidia-nvfx  # 或按官方指引从 NVIDIA 资源下载
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import config
from src import ffmpeg_utils, img_sr
from src.common import (
    cleanup_temp_dir,
    clear_cuda_cache,
    get_logger,
    make_temp_subdir,
    safe_remove,
    unique_output_path,
)

logger = get_logger("maxine")

# --------------------------------------------------------------------------- #
# 可选依赖探测
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - 依赖运行环境
    import nvvfx

    HAS_MAXINE = True
    MAXINE_IMPORT_ERROR = ""
except Exception as exc:  # noqa: BLE001
    nvvfx = None  # type: ignore[assignment]
    HAS_MAXINE = False
    MAXINE_IMPORT_ERROR = str(exc)


class MaxineError(RuntimeError):
    """Maxine VSR 相关的可读错误。"""


def maxine_status() -> str:
    """返回 Maxine 可用性文案（供 UI 展示）。"""
    if HAS_MAXINE:
        return "✅ Maxine VSR 可用（闭源组件，不支持 LoRA 微调）"
    return (
        "ℹ️ 未检测到 Maxine VSR（nvvfx），已自动隐藏该选项。\n"
        "如需使用，请按 NVIDIA 官方文档安装 nvidia-vfx / nvvfx 相关组件。\n"
        f"导入错误：{MAXINE_IMPORT_ERROR or '未安装'}"
    )


def _require_maxine() -> None:
    if not HAS_MAXINE:
        raise MaxineError(
            "Maxine VSR 不可用（未安装 nvvfx）。\n" + maxine_status()
        )


# --------------------------------------------------------------------------- #
# 单帧时序超分封装
# --------------------------------------------------------------------------- #
class MaxineVSR:
    """Maxine VideoSuperRes 的薄封装：一次调用处理一帧（按顺序喂帧即有时序增强）。

    由于不同 nvvfx 版本 API 存在差异，本类对字段/方法做了存在性检查，
    并在不匹配时给出明确的版本提示，而不是抛出难以理解的底层异常。
    """

    def __init__(
        self,
        *,
        input_width: int,
        input_height: int,
        output_width: int,
        output_height: int,
        device: str = "cuda:0",
    ):
        _require_maxine()
        self.device = device
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.output_width = int(output_width)
        self.output_height = int(output_height)
        self._vsr = None
        self._torch = None

    def load(self) -> MaxineVSR:
        """创建并初始化 VSR 会话。"""
        try:
            self._vsr = nvvfx.VideoSuperRes()
        except Exception as exc:  # noqa: BLE001
            raise MaxineError(f"无法创建 Maxine VSR 会话：{exc}") from exc

        for attribute, value in (
            ("input_width", self.input_width),
            ("input_height", self.input_height),
            ("output_width", self.output_width),
            ("output_height", self.output_height),
        ):
            if hasattr(self._vsr, attribute):
                try:
                    setattr(self._vsr, attribute, value)
                except Exception as exc:  # noqa: BLE001
                    raise MaxineError(f"设置 {attribute} 失败：{exc}") from exc

        if hasattr(self._vsr, "device"):
            try:
                self._vsr.device = self.device
            except Exception:  # noqa: BLE001
                logger.debug("该版本 nvvfx 不支持设置 device，忽略。")

        load_method = getattr(self._vsr, "load", None)
        if callable(load_method):
            try:
                load_method()
            except Exception as exc:  # noqa: BLE001
                raise MaxineError(f"Maxine VSR 初始化失败：{exc}") from exc

        try:
            import torch

            self._torch = torch
        except Exception:  # noqa: BLE001
            self._torch = None

        logger.info(
            "Maxine VSR 就绪：输入 %dx%d -> 输出 %dx%d",
            self.input_width, self.input_height, self.output_width, self.output_height,
        )
        return self

    def upscale_frame(self, frame_rgb):
        """对单帧做时序超分，返回同尺寸比例的 RGB 数组。

        注意：帧必须**按时间顺序**依次传入，否则时序增强效果会异常。
        """
        if self._vsr is None:
            raise MaxineError("Maxine VSR 会话尚未初始化，请先调用 load()。")
        if frame_rgb is None or getattr(frame_rgb, "size", 0) == 0:
            raise MaxineError("输入帧为空。")

        payload = frame_rgb
        # 部分版本接受 GPU 上的 torch 张量，优先使用以避免额外拷贝
        if self._torch is not None and hasattr(self._vsr, "run"):
            try:
                tensor = self._torch.from_numpy(frame_rgb).to(self.device, non_blocking=True)
                payload = tensor
            except Exception:  # noqa: BLE001
                payload = frame_rgb

        try:
            output = self._vsr.run(payload)
        except TypeError:
            # 老版本 API 需要显式传入尺寸
            try:
                output = self._vsr.run(payload, self.output_width, self.output_height)
            except Exception as exc:  # noqa: BLE001
                raise MaxineError(f"Maxine VSR 推理失败：{exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise MaxineError(f"Maxine VSR 推理失败：{exc}") from exc

        if output is None:
            raise MaxineError("Maxine VSR 返回空结果。")
        if hasattr(output, "detach"):  # torch 张量
            output = output.detach().to("cpu").numpy()
        return output

    def release(self) -> None:
        """释放会话资源。"""
        self._vsr = None
        clear_cuda_cache()


# --------------------------------------------------------------------------- #
# 视频链路
# --------------------------------------------------------------------------- #
def _output_size(width: int, height: int, preset_name: str | None, max_scale: float = 4.0) -> tuple[int, int]:
    """按分辨率预设计算 Maxine 输出尺寸（长边受限，保持宽高比）。"""
    long_edge = img_sr.resolve_target_long_edge(preset_name)
    if long_edge <= 0:
        scale = 2.0
    else:
        scale = long_edge / float(max(width, height))
    scale = max(1.0, min(max_scale, scale))
    out_w = int(round(width * scale / 2) * 2)  # 保持偶数，避免编码器不满
    out_h = int(round(height * scale / 2) * 2)
    return max(2, out_w), max(2, out_h)


def upscale_video_maxine(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    preset_name: str | None = None,
    device: str = "cuda:0",
    video_format: str = config.DEFAULT_VIDEO_FORMAT,
    preview_only: bool = False,
    preview_frames: int = config.PREVIEW_FRAME_LIMIT,
    fps_override: float | None = None,
    bitrate_mbps: float | None = None,
    crf: int | None = None,
    keep_audio: bool = True,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[Path, dict]:
    """用 Maxine VSR 对视频做时序超分（逐帧顺序推理）。

    与 RealESRGAN 链路一样，全程在 temp/ 子目录中操作并在结束时清理。
    """
    _require_maxine()
    started = time.time()
    ffmpeg_utils.ensure_ffmpeg()

    source = Path(input_path)
    if not source.is_file():
        raise MaxineError(f"输入视频不存在：{source}")

    def emit(value: float, message: str) -> None:
        if progress:
            try:
                progress(max(0.0, min(1.0, value)), message)
            except Exception:  # noqa: BLE001
                logger.debug("进度回调异常（已忽略）：%s", message)

    emit(0.01, "读取视频信息…")
    try:
        info = ffmpeg_utils.get_video_info(source)
    except ffmpeg_utils.FFmpegError as exc:
        raise MaxineError(f"无法读取视频信息，文件可能已损坏：{exc}") from exc

    src_fps = float(fps_override or info.get("fps") or config.DEFAULT_FPS)
    in_w, in_h = int(info.get("width") or 640), int(info.get("height") or 360)
    out_w, out_h = _output_size(in_w, in_h, preset_name)

    work_dir = make_temp_subdir("maxine_sr")
    frames_dir = work_dir / "frames"
    out_frames_dir = work_dir / "frames_sr"
    audio_ext = config.VIDEO_CODEC_MAP.get(video_format.upper(), {}).get("audio_ext", "m4a")
    audio_path = work_dir / f"audio.{audio_ext}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_frames_dir.mkdir(parents=True, exist_ok=True)

    vsr: MaxineVSR | None = None
    try:
        emit(0.04, "初始化 Maxine VSR…")
        vsr = MaxineVSR(
            input_width=in_w,
            input_height=in_h,
            output_width=out_w,
            output_height=out_h,
            device=device,
        ).load()

        emit(0.10, "拆分视频帧…")
        frames = ffmpeg_utils.extract_frames(
            source,
            frames_dir,
            max_frames=int(preview_frames) if preview_only else None,
            image_format="PNG",
        )
        total = len(frames)

        processed = 0
        for index, frame_path in enumerate(frames, start=1):
            frame = img_sr.read_image(frame_path)
            try:
                result = vsr.upscale_frame(frame)
                img_sr.write_image(out_frames_dir / frame_path.name, result)
                processed += 1
            except MaxineError as exc:
                logger.warning("第 %d 帧 Maxine 处理失败，使用原帧代替：%s", index, exc)
                img_sr.write_image(out_frames_dir / frame_path.name, frame)
            if index % 5 == 0 or index == total:
                ratio = index / max(1, total)
                emit(0.12 + ratio * 0.78, f"Maxine 超分中 {index}/{total} 帧（{ratio * 100:.0f}%）")

        if processed == 0:
            raise MaxineError("没有任何一帧成功超分，请检查输入视频或输出尺寸设置。")

        if keep_audio and info.get("has_audio"):
            emit(0.91, "提取音轨…")
            try:
                ffmpeg_utils.extract_audio(
                    source, audio_path, audio_codec="copy" if video_format.upper() == "MKV" else "aac"
                )
            except ffmpeg_utils.FFmpegError as exc:
                logger.warning("音频提取失败，将输出无声视频：%s", exc)

        emit(0.94, "合成输出视频…")
        if output_path is None:
            suffix = ".mkv" if video_format.upper() == "MKV" else ".mp4"
            stem = f"{source.stem}_maxine" + ("_preview" if preview_only else "")
            output_path = unique_output_path(stem, suffix)
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            ffmpeg_utils.combine_frames_to_video(
                out_frames_dir,
                target,
                fps=src_fps,
                audio_path=audio_path if audio_path.exists() else None,
                video_format=video_format,
                bitrate_mbps=bitrate_mbps,
                crf=crf,
            )
        except ffmpeg_utils.FFmpegError as exc:
            raise MaxineError(f"视频合成失败：{exc}") from exc

        elapsed = time.time() - started
        result_info = {
            "elapsed": elapsed,
            "frames": processed,
            "preview": preview_only,
            "input_size": f"{in_w}x{in_h}",
            "output_size": f"{out_w}x{out_h}",
            "fps": src_fps,
            "duration": round(processed / src_fps, 2),
            "audio": "保留" if audio_path.exists() else "无",
            "model": "Maxine VSR (时序)",
            "lora": "不支持（闭源组件）",
            "output_path": str(target),
        }
        emit(1.0, f"完成：{target.name}")
        logger.info("Maxine 视频超分完成：%d 帧 / %.2fs -> %s", processed, elapsed, target)
        return target, result_info

    finally:
        if vsr is not None:
            vsr.release()
        if config.KEEP_TEMP_FILES:
            logger.info("保留临时目录用于排查：%s", work_dir)
        else:
            safe_remove(work_dir)
            cleanup_temp_dir()


__all__ = [
    "HAS_MAXINE",
    "MaxineError",
    "MaxineVSR",
    "maxine_status",
    "upscale_video_maxine",
]
