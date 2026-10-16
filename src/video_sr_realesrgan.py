"""RealESRGAN 逐帧视频超分。

流程：``ffprobe 探测 -> ffmpeg 拆帧 -> 逐帧超分 -> 提取音轨 -> 合成视频``，
全程只在 temp/ 下的独立子目录中操作，结束（无论成功失败）都会清理。
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

logger = get_logger("video_sr")


class VideoSRError(RuntimeError):
    """视频超分过程中的可读错误。"""


def estimate_required_vram_gb(target_long_edge: int = 0, outscale: float = 4.0) -> float:
    """逐帧视频超分的显存估算（与单图一致，逐帧不会叠加占用）。"""
    return img_sr.estimate_required_vram_gb(target_long_edge, outscale)


def _emit(progress: Callable[[float, str], None] | None, value: float, message: str) -> None:
    if progress:
        try:
            progress(max(0.0, min(1.0, value)), message)
        except Exception:  # noqa: BLE001 - 进度回调失败不应中断任务
            logger.debug("进度回调异常（已忽略）：%s", message)


def upscale_video(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    preset_name: str | None = None,
    device: str = "cuda:0",
    model_name: str = config.DEFAULT_MODEL_NAME,
    lora_selection: str | None = None,
    half: bool | None = None,
    outscale: float = 4.0,
    tile: int | None = None,
    auto_download: bool = True,
    video_format: str = config.DEFAULT_VIDEO_FORMAT,
    preview_only: bool = False,
    preview_frames: int = config.PREVIEW_FRAME_LIMIT,
    fps_override: float | None = None,
    bitrate_mbps: float | None = None,
    crf: int | None = None,
    keep_audio: bool = True,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[Path, dict]:
    """对视频做 RealESRGAN 逐帧超分。

    Args:
        preview_only: 只渲染前 ``preview_frames`` 帧，快速看效果（P1）。
        fps_override: 覆盖输出帧率（None 表示沿用源帧率）。
        bitrate_mbps: 目标码率；None 时使用 CRF/CQ 质量控制。

    Returns:
        ``(输出视频路径, 信息字典)``
    """
    started = time.time()
    ffmpeg_utils.ensure_ffmpeg()

    source = Path(input_path)
    if not source.is_file():
        raise VideoSRError(f"输入视频不存在：{source}")

    # ---------- 1. 探测源信息 ----------
    _emit(progress, 0.01, "读取视频信息…")
    try:
        info = ffmpeg_utils.get_video_info(source)
    except ffmpeg_utils.FFmpegError as exc:
        raise VideoSRError(f"无法读取视频信息，文件可能已损坏：{exc}") from exc

    src_fps = float(fps_override or info.get("fps") or config.DEFAULT_FPS)
    target_long_edge = img_sr.resolve_target_long_edge(preset_name)
    image_format = "PNG"

    # ---------- 2. 准备临时目录 ----------
    work_dir = make_temp_subdir("video_sr")
    frames_dir = work_dir / "frames"
    out_frames_dir = work_dir / "frames_sr"
    audio_path = work_dir / f"audio.{config.VIDEO_CODEC_MAP.get(video_format.upper(), {}).get('audio_ext', 'm4a')}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ---------- 3. 加载模型（在拆帧前完成，尽早暴露权重/显存问题） ----------
        _emit(progress, 0.03, "加载超分模型…")
        engine = img_sr.get_engine().load(
            model_name=model_name,
            device=device,
            half=half,
            tile=tile,
            lora_selection=lora_selection,
            auto_download=auto_download,
            progress=lambda value, message: _emit(progress, 0.03 + value * 0.07, message),
        )

        # ---------- 4. 拆帧 ----------
        _emit(progress, 0.10, "拆分视频帧…")
        frames = ffmpeg_utils.extract_frames(
            source,
            frames_dir,
            max_frames=int(preview_frames) if preview_only else None,
            image_format=image_format,
        )
        total = len(frames)
        logger.info("待处理帧数：%d（预览模式=%s）", total, preview_only)

        # ---------- 5. 逐帧超分 ----------
        processed = 0
        for index, frame_path in enumerate(frames, start=1):
            try:
                frame = img_sr.read_image(frame_path)
                result = engine.upscale(frame, outscale=outscale, target_long_edge=target_long_edge)
                img_sr.write_image(out_frames_dir / frame_path.name.replace(".jpg", ".png"), result)
                processed += 1
            except img_sr.ImageSRError as exc:
                # 单帧失败不中断整片，用原帧顶替，保证时长一致
                logger.warning("第 %d 帧处理失败，使用原帧代替：%s", index, exc)
                try:
                    img_sr.write_image(
                        out_frames_dir / frame_path.name.replace(".jpg", ".png"),
                        img_sr.read_image(frame_path),
                    )
                except Exception:  # noqa: BLE001
                    continue
            if index % 5 == 0 or index == total:
                ratio = index / max(1, total)
                _emit(
                    progress,
                    0.12 + ratio * 0.78,
                    f"超分处理中 {index}/{total} 帧（{ratio * 100:.0f}%）",
                )

        if processed == 0:
            raise VideoSRError("没有任何一帧成功超分，请检查输入视频或降低分辨率后重试。")

        # ---------- 6. 提取音轨 ----------
        if keep_audio and info.get("has_audio"):
            _emit(progress, 0.91, "提取音轨…")
            try:
                ffmpeg_utils.extract_audio(
                    source, audio_path, audio_codec="copy" if video_format.upper() == "MKV" else "aac"
                )
            except ffmpeg_utils.FFmpegError as exc:
                logger.warning("音频提取失败，将输出无声视频：%s", exc)

        # ---------- 7. 合成视频 ----------
        _emit(progress, 0.94, "合成输出视频…")
        if output_path is None:
            suffix = ".mkv" if video_format.upper() == "MKV" else ".mp4"
            stem = f"{source.stem}_sr_preview" if preview_only else f"{source.stem}_sr"
            output_path = unique_output_path(stem, suffix)
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            ffmpeg_utils.combine_frames_to_video(
                out_frames_dir,
                target,
                fps=src_fps,
                audio_path=audio_path if audio_path.exists() else None,
                image_format=image_format,
                video_format=video_format,
                bitrate_mbps=bitrate_mbps,
                crf=crf,
            )
        except ffmpeg_utils.FFmpegError as exc:
            raise VideoSRError(f"视频合成失败：{exc}") from exc

        elapsed = time.time() - started
        out_w = info.get("width")
        out_h = info.get("height")
        if out_w and out_h and target_long_edge:
            scale = target_long_edge / float(max(out_w, out_h))
            out_w, out_h = int(round(out_w * scale)), int(round(out_h * scale))
        elif out_w and out_h:
            out_w, out_h = int(out_w * outscale), int(out_h * outscale)

        result_info = {
            "elapsed": elapsed,
            "frames": processed,
            "preview": preview_only,
            "input_size": f"{info.get('width')}x{info.get('height')}",
            "output_size": f"{out_w}x{out_h}",
            "fps": src_fps,
            "duration": round(processed / src_fps, 2),
            "audio": "保留" if audio_path.exists() else "无",
            "model": model_name,
            "lora": Path(engine.lora_name).name if engine.lora_name else "不使用 LoRA",
            "output_path": str(target),
        }
        _emit(progress, 1.0, f"完成：{target.name}")
        logger.info(
            "视频超分完成：%d 帧 / %.2fs -> %s（模型=%s，LoRA=%s）",
            processed, elapsed, target, model_name, result_info["lora"],
        )
        clear_cuda_cache()
        return target, result_info

    finally:
        # 无论成功失败都清理临时产物（可被 config.KEEP_TEMP_FILES 关闭）
        if config.KEEP_TEMP_FILES:
            logger.info("保留临时目录用于排查：%s", work_dir)
        else:
            safe_remove(work_dir)
            cleanup_temp_dir()


__all__ = ["VideoSRError", "upscale_video", "estimate_required_vram_gb"]
