"""RealESRGAN 逐帧视频超分。

流程：``ffprobe 探测 -> ffmpeg 拆帧 -> 逐帧超分 -> 提取音轨 -> 合成视频``，
全程只在 temp/ 下的独立子目录中操作，结束（无论成功失败）都会清理。
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

import config
from src import ffmpeg_utils, img_sr, task_control
from src.common import (
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


def _upscale_video_by_files(
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
    should_cancel: Callable[[], None] | None = None,
) -> tuple[Path, dict]:
    """对视频做 RealESRGAN 逐帧超分。

    Args:
        preview_only: 只渲染前 ``preview_frames`` 帧，快速看效果（P1）。
        fps_override: 覆盖输出帧率（None 表示沿用源帧率）。
        bitrate_mbps: 目标码率；None 时使用 CRF/CQ 质量控制。
        should_cancel: 可选的取消检查回调，**每帧之前**调用一次；
            被取消时应抛出 ``TaskCancelled``，此时 ``finally`` 会清理临时目录。

    Returns:
        ``(输出视频路径, 信息字典)``
    """
    started = time.time()
    ffmpeg_utils.ensure_ffmpeg()

    # UI 可能直接传下拉标签（"cuda:0 | <名称> (32 GB, sm_89)"），先归一化，
    # 后续交给引擎与日志的值才一致、才不会被 torch 拒绝。
    device = config.normalize_device(device)

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
        if should_cancel is not None:
            should_cancel()
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
            # 每帧之前检查一次，并把检查传给引擎：后者会在块级 forward 前钩子里
            # 再检查几十次，所以长视频的取消不必等整帧跑完。
            if should_cancel is not None:
                should_cancel()
            try:
                frame = img_sr.read_image(frame_path)
                result = engine.upscale(
                    frame,
                    outscale=outscale,
                    target_long_edge=target_long_edge,
                    should_cancel=should_cancel,
                )
                img_sr.write_image(out_frames_dir / frame_path.name.replace(".jpg", ".png"), result)
                processed += 1
            except task_control.TaskCancelled:
                raise
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
            "pipeline": "files",
        }
        _emit(progress, 1.0, f"完成：{target.name}")
        logger.info(
            "视频超分完成：%d 帧 / %.2fs -> %s（模型=%s，LoRA=%s）",
            processed, elapsed, target, model_name, result_info["lora"],
        )
        clear_cuda_cache()
        return target, result_info

    finally:
        # 只清理本次任务自己的子目录（可被 config.KEEP_TEMP_FILES 关闭）。
        # 这里**不能**调用 cleanup_temp_dir()：它会清空整个 temp/，而
        # Gradio 队列默认并发 4，两个视频任务同时跑时会互相删掉对方的帧目录。
        # 全局清理只应发生在启动阶段（见 src.common.cleanup_stale_temp）。
        if config.KEEP_TEMP_FILES:
            logger.info("保留临时目录用于排查：%s", work_dir)
        else:
            safe_remove(work_dir)


def _log_tail(path: Path, lines: int = 6) -> str:
    """读取 ffmpeg 日志尾部，用于把错误转成可读提示。"""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def _streaming_enabled() -> bool:
    """是否启用管道式管线（``PVSR_VIDEO_PIPELINE=files`` 可强制回退）。"""
    return config.video_pipeline() == "stream"


def _upscale_video_streaming(
    source: Path,
    info: dict,
    *,
    output_path: str | Path | None,
    preset_name: str | None,
    device: str,
    model_name: str,
    lora_selection: str | None,
    half: bool | None,
    outscale: float,
    tile: int | None,
    auto_download: bool,
    video_format: str,
    preview_only: bool,
    preview_frames: int,
    fps_override: float | None,
    bitrate_mbps: float | None,
    crf: int | None,
    keep_audio: bool,
    progress: Callable[[float, str], None] | None,
    should_cancel: Callable[[], None] | None,
    started: float,
) -> tuple[Path, dict]:
    """流式管线：解码 → 推理 → 编码**全程走管道**，不落任何中间帧文件。

    与文件管线相比省掉了每帧一次的「PNG 编码 + 落盘 + 读回 + 解码」，实测 640x360
    输入每帧约省 180 ms（占单帧总耗时的三成多），GPU 也因此不再在 IO 上干等；
    拆帧阶段整段消失，解码与推理天然重叠。

    失败时抛异常，由 :func:`upscale_video` 自动回退到文件管线。
    """
    import numpy as np

    src_fps = float(fps_override or info.get("fps") or config.DEFAULT_FPS)
    in_w = int(info.get("width") or 0)
    in_h = int(info.get("height") or 0)
    if in_w <= 0 or in_h <= 0:
        raise VideoSRError("无法确定源视频分辨率，流式管线无法继续。")
    frame_bytes = in_w * in_h * 3
    target_long_edge = img_sr.resolve_target_long_edge(preset_name)
    expected_total = int(info.get("nb_frames") or 0) or None

    work_dir = make_temp_subdir("video_sr_stream")
    audio_ext = config.VIDEO_CODEC_MAP.get(
        video_format.upper(), {}
    ).get("audio_ext", "m4a")
    audio_path = work_dir / f"audio.{audio_ext}"
    decode_log = work_dir / "decode.log"
    encode_log = work_dir / "encode.log"

    decoder = None
    encoder = None
    processed = 0
    out_w = out_h = 0
    succeeded = False
    try:
        _emit(progress, 0.02, "加载超分模型…")
        engine = img_sr.get_engine().load(
            model_name=model_name,
            device=device,
            half=half,
            tile=tile,
            lora_selection=lora_selection,
            auto_download=auto_download,
            progress=lambda value, message: _emit(progress, 0.02 + value * 0.06, message),
        )

        if keep_audio and info.get("has_audio"):
            _emit(progress, 0.09, "提取音轨…")
            try:
                ffmpeg_utils.extract_audio(
                    source,
                    audio_path,
                    audio_codec="copy" if video_format.upper() == "MKV" else "aac",
                )
            except ffmpeg_utils.FFmpegError as exc:
                logger.warning("音频提取失败，将输出无声视频：%s", exc)

        if output_path is None:
            suffix = ".mkv" if video_format.upper() == "MKV" else ".mp4"
            stem = f"{source.stem}_sr_preview" if preview_only else f"{source.stem}_sr"
            output_path = unique_output_path(stem, suffix)
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        _emit(progress, 0.10, "解码 + 超分 + 编码（流式）…")
        with open(decode_log, "wb") as decode_err, open(encode_log, "wb") as encode_err:
            decoder = ffmpeg_utils.spawn_frame_decoder(source, stderr=decode_err)
            try:
                while True:
                    if should_cancel is not None:
                        should_cancel()
                    raw = ffmpeg_utils.read_exact(decoder.stdout, frame_bytes)
                    if raw is None:
                        break  # 正常结束
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(in_h, in_w, 3)
                    result = engine.upscale(
                        frame,
                        outscale=outscale,
                        target_long_edge=target_long_edge,
                        should_cancel=should_cancel,
                    )

                    if encoder is None:
                        # 以第一帧的实际输出尺寸启动编码器（含 yuv420p 的偶数约束）
                        out_h, out_w = result.shape[:2]
                        out_w = ffmpeg_utils.even_size(out_w)
                        out_h = ffmpeg_utils.even_size(out_h)
                        encoder = ffmpeg_utils.spawn_frame_encoder(
                            target,
                            out_w,
                            out_h,
                            fps=src_fps,
                            stderr=encode_err,
                            audio_path=audio_path if audio_path.exists() else None,
                            video_format=video_format,
                            bitrate_mbps=bitrate_mbps,
                            crf=crf,
                        )
                    elif (result.shape[1], result.shape[0]) != (out_w, out_h):
                        import cv2  # 仅尺寸兜底时用到

                        result = cv2.resize(
                            result, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4
                        )

                    try:
                        encoder.stdin.write(np.ascontiguousarray(result).tobytes())
                    except (BrokenPipeError, ValueError) as exc:
                        raise VideoSRError(
                            "编码进程提前退出，无法继续写入帧。\n"
                            + _log_tail(encode_log)
                        ) from exc

                    processed += 1
                    if preview_only and processed >= int(preview_frames):
                        break
                    if expected_total:
                        ratio = min(1.0, processed / expected_total)
                        _emit(
                            progress,
                            0.12 + ratio * 0.80,
                            f"超分处理中 {processed}/{expected_total} 帧（{ratio * 100:.0f}%）",
                        )
                    elif processed % 5 == 0:
                        _emit(progress, 0.5, f"超分处理中 {processed} 帧…")
            finally:
                if decoder is not None:
                    if decoder.poll() is None:
                        decoder.terminate()
                        try:
                            decoder.wait(timeout=10)
                        except Exception:  # noqa: BLE001
                            decoder.kill()
                    if decoder.stdout is not None:
                        decoder.stdout.close()

            if encoder is None:
                raise VideoSRError(
                    "没有任何一帧被成功解码，请确认视频未损坏。\n" + _log_tail(decode_log)
                )

            _emit(progress, 0.95, "收尾编码…")
            try:
                encoder.stdin.close()
            except Exception:  # noqa: BLE001 - 编码器可能已退出
                pass
            encode_code = encoder.wait()
            if encode_code != 0 or not target.is_file():
                raise VideoSRError(
                    "视频编码失败（退出码 "
                    f"{encode_code}）。\n" + _log_tail(encode_log)
                )

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
            "model": model_name,
            "lora": Path(engine.lora_name).name if engine.lora_name else "不使用 LoRA",
            "output_path": str(target),
            "pipeline": "stream",
        }
        _emit(progress, 1.0, f"完成：{target.name}")
        logger.info(
            "视频超分完成（流式）：%d 帧 / %.2fs -> %s（%.1f 帧/s）",
            processed, elapsed, target, processed / max(elapsed, 1e-6),
        )
        clear_cuda_cache()
        succeeded = True
        return target, result_info

    finally:
        # 任何异常路径都要收掉编码进程，否则它会一直等着 stdin（Windows 上还会占住输出文件）
        if encoder is not None and encoder.poll() is None:
            try:
                if encoder.stdin is not None:
                    encoder.stdin.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                encoder.terminate()
                encoder.wait(timeout=10)
            except Exception:  # noqa: BLE001
                encoder.kill()
        # 出错时不留半成品文件
        if "target" in locals() and target.exists() and not succeeded:
            target.unlink(missing_ok=True)
        if config.KEEP_TEMP_FILES:
            logger.info("保留临时目录用于排查：%s", work_dir)
        else:
            safe_remove(work_dir)


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
    should_cancel: Callable[[], None] | None = None,
) -> tuple[Path, dict]:
    """对视频做 RealESRGAN 逐帧超分。

    默认走**管道式管线**（解码 → 推理 → 编码全程流式，不落中间帧文件）；
    遇到环境不支持等异常时自动回退到「拆帧 → 逐帧 → 合成」的文件式管线。
    两种管线的参数与输出语义完全一致。

    Args:
        preview_only: 只渲染前 ``preview_frames`` 帧，快速看效果。
        fps_override: 覆盖输出帧率（None 表示沿用源帧率）。
        bitrate_mbps: 目标码率；None 时使用 CRF/CQ 质量控制。
        should_cancel: 可选的取消检查回调，**每帧之前**调用一次；
            被取消时应抛出 ``TaskCancelled``，此时临时目录会被清理。

    Returns:
        ``(输出视频路径, 信息字典)``；``info["pipeline"]`` 标明实际使用的管线。
    """
    started = time.time()
    ffmpeg_utils.ensure_ffmpeg()

    # UI 可能直接传下拉标签（"cuda:0 | <名称> (32 GB, sm_89)"），先归一化
    device = config.normalize_device(device)

    source = Path(input_path)
    if not source.is_file():
        raise VideoSRError(f"输入视频不存在：{source}")

    common = dict(
        output_path=output_path,
        preset_name=preset_name,
        device=device,
        model_name=model_name,
        lora_selection=lora_selection,
        half=half,
        outscale=outscale,
        tile=tile,
        auto_download=auto_download,
        video_format=video_format,
        preview_only=preview_only,
        preview_frames=preview_frames,
        fps_override=fps_override,
        bitrate_mbps=bitrate_mbps,
        crf=crf,
        keep_audio=keep_audio,
        progress=progress,
        should_cancel=should_cancel,
    )

    if _streaming_enabled():
        try:
            info = ffmpeg_utils.get_video_info(source)
        except ffmpeg_utils.FFmpegError as exc:
            raise VideoSRError(f"无法读取视频信息，文件可能已损坏：{exc}") from exc
        try:
            return _upscale_video_streaming(source, info, started=started, **common)
        except task_control.TaskCancelled:
            raise
        except VideoSRError:
            raise
        except Exception as exc:  # noqa: BLE001 - 任何异常都退回文件管线
            logger.warning("管道式管线不可用，已回退到文件管线：%s", exc)

    return _upscale_video_by_files(input_path, **common)


__all__ = ["VideoSRError", "upscale_video", "estimate_required_vram_gb"]
