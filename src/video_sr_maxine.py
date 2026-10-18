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

    # 只 import 成功还不够：conda 卸载后 site-packages 里可能残留空的 nvvfx/
    # 目录，会被当作命名空间包导入成功，直到真正调用才报
    # ``module 'nvvfx' has no attribute 'VideoSuperRes'``。
    if not hasattr(nvvfx, "VideoSuperRes"):
        raise ImportError(
            "已安装的 nvvfx 缺少 VideoSuperRes（可能是残留目录或版本不匹配，"
            "请用 python scripts/install.py --maxine 重装）"
        )
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
        self.device = config.normalize_device(device)
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.output_width = int(output_width)
        self.output_height = int(output_height)
        self._vsr = None
        self._torch = None

    def load(self) -> MaxineVSR:
        """创建并初始化 VSR 会话。"""
        try:
            import torch

            self._torch = torch
        except Exception as exc:  # noqa: BLE001
            raise MaxineError(
                f"Maxine VSR 需要 PyTorch 来完成张量转换：{exc}"
            ) from exc

        # nvidia-vfx 0.2.x 的构造签名：
        #   VideoSuperRes(quality, strength, device, image_encoding)
        # 老版本没有这些参数，逐个尝试以便给出可读的版本提示。
        kwargs: dict = {"device": config.gpu_index_from_option(self.device)}
        quality = getattr(nvvfx, "QualityLevel", None)
        if quality is not None:
            kwargs["quality"] = getattr(quality, "HIGH", None) or quality
        try:
            self._vsr = nvvfx.VideoSuperRes(**kwargs)
        except TypeError:
            kwargs.pop("quality", None)
            try:
                self._vsr = nvvfx.VideoSuperRes(**kwargs)
            except Exception as exc:  # noqa: BLE001
                raise MaxineError(f"无法创建 Maxine VSR 会话：{exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise MaxineError(f"无法创建 Maxine VSR 会话：{exc}") from exc

        # 0.2.x 只需要输出尺寸，输入尺寸由每帧推断；老版本可能还有 input_* 字段
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

        load_method = getattr(self._vsr, "load", None)
        if callable(load_method):
            try:
                load_method()
            except Exception as exc:  # noqa: BLE001
                raise MaxineError(f"Maxine VSR 初始化失败：{exc}") from exc

        logger.info(
            "Maxine VSR 就绪：输出 %dx%d（nvvfx %s）",
            self.output_width, self.output_height, getattr(nvvfx, "__version__", "?"),
        )
        return self

    def upscale_frame(self, frame_rgb):
        """对单帧做时序超分，返回 RGB uint8 数组（HWC）。

        注意：帧必须**按时间顺序**依次传入，否则时序增强效果会异常。

        nvidia-vfx 0.2.x 的约定是「CUDA 上 channels-first 的 float32 张量、
        取值 [0,1]」，``run()`` 返回带 DLPack ``.image`` 的对象；这里负责
        numpy ↔ torch ↔ DLPack 的全部转换。
        """
        if self._vsr is None:
            raise MaxineError("Maxine VSR 会话尚未初始化，请先调用 load()。")
        if frame_rgb is None or getattr(frame_rgb, "size", 0) == 0:
            raise MaxineError("输入帧为空。")

        torch = self._torch
        frame = frame_rgb
        if frame.ndim == 2:  # 灰度兜底
            frame = frame[:, :, None].repeat(3, axis=2)
        if frame.shape[2] == 4:  # 丢弃 alpha
            frame = frame[:, :, :3]
        # 管道式解码拿到的数组是只读的（来自 bytes），torch 不接受不可写张量
        if not getattr(frame, "flags", None) or not frame.flags.writeable:
            frame = frame.copy()

        # HWC uint8 RGB -> CHW float32 [0,1] 连续张量（contiguous 是硬要求）
        tensor = (
            torch.from_numpy(frame)
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .div_(255.0)
            .contiguous()
            .to(self.device)
        )

        try:
            result = self._vsr.run(tensor)
        except Exception as exc:  # noqa: BLE001
            raise MaxineError(f"Maxine VSR 推理失败：{exc}") from exc

        return self._to_rgb_array(result)

    def _to_rgb_array(self, result):
        """把 ``run()`` 的返回值转成 HWC uint8 RGB 数组。"""
        torch = self._torch
        image = getattr(result, "image", result)

        if torch is not None and hasattr(torch, "from_dlpack") and not torch.is_tensor(image):
            try:
                # DLPack 指向内部缓冲：必须 clone，否则下一次 run() 会覆盖
                tensor = torch.from_dlpack(image).clone()
            except Exception as exc:  # noqa: BLE001
                raise MaxineError(f"无法解析 Maxine 返回的 DLPack 结果：{exc}") from exc
        elif torch is not None and torch.is_tensor(image):
            tensor = image.detach().clone()
        else:
            raise MaxineError(
                f"无法解析 Maxine 返回结果（类型 {type(result).__name__}），"
                "可能需要更新 nvidia-vfx。"
            )

        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise MaxineError(f"Maxine 输出维度异常：{tuple(tensor.shape)}")

        array = (
            tensor.detach()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
        )
        return array

    def release(self) -> None:
        """释放会话资源。"""
        vsr, self._vsr = self._vsr, None
        if vsr is not None:
            for name in ("close", "release"):
                method = getattr(vsr, name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:  # noqa: BLE001
                        logger.debug("Maxine 会话关闭失败（忽略）")
                    break
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


def _maxine_streaming(
    source: Path,
    info: dict,
    *,
    output_path: str | Path | None,
    preset_name: str | None,
    device: str,
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
    """Maxine 的管道式管线：解码 → VSR → 编码，全程不落中间帧。

    Maxine 单帧只要 ~10 ms，文件管线的 PNG 编码/落盘反而是主要开销，
    所以这里同样走 ``rawvideo`` 管道。
    """
    import numpy as np

    def emit(value: float, message: str) -> None:
        if progress:
            try:
                progress(max(0.0, min(1.0, value)), message)
            except Exception:  # noqa: BLE001
                logger.debug("进度回调异常（已忽略）：%s", message)

    src_fps = float(fps_override or info.get("fps") or config.DEFAULT_FPS)
    in_w = int(info.get("width") or 0)
    in_h = int(info.get("height") or 0)
    if in_w <= 0 or in_h <= 0:
        raise MaxineError("无法确定源视频分辨率，流式管线无法继续。")
    frame_bytes = in_w * in_h * 3
    out_w, out_h = _output_size(in_w, in_h, preset_name)
    expected_total = int(info.get("nb_frames") or 0) or None

    work_dir = make_temp_subdir("maxine_sr_stream")
    audio_ext = config.VIDEO_CODEC_MAP.get(
        video_format.upper(), {}
    ).get("audio_ext", "m4a")
    audio_path = work_dir / f"audio.{audio_ext}"
    decode_log = work_dir / "decode.log"
    encode_log = work_dir / "encode.log"

    vsr = None
    decoder = None
    encoder = None
    processed = 0
    succeeded = False
    target: Path | None = None
    try:
        emit(0.04, "初始化 Maxine VSR…")
        vsr = MaxineVSR(
            input_width=in_w,
            input_height=in_h,
            output_width=out_w,
            output_height=out_h,
            device=device,
        ).load()

        if keep_audio and info.get("has_audio"):
            emit(0.10, "提取音轨…")
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
            stem = f"{source.stem}_maxine" + ("_preview" if preview_only else "")
            output_path = unique_output_path(stem, suffix)
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        emit(0.12, "解码 + Maxine 超分 + 编码（流式）…")
        with open(decode_log, "wb") as decode_err, open(encode_log, "wb") as encode_err:
            decoder = ffmpeg_utils.spawn_frame_decoder(source, stderr=decode_err)
            try:
                while True:
                    if should_cancel is not None:
                        should_cancel()
                    raw = ffmpeg_utils.read_exact(decoder.stdout, frame_bytes)
                    if raw is None:
                        break
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(in_h, in_w, 3)
                    result = vsr.upscale_frame(frame)

                    if (result.shape[1], result.shape[0]) != (out_w, out_h):
                        import cv2

                        result = cv2.resize(
                            result, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4
                        )

                    if encoder is None:
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

                    try:
                        encoder.stdin.write(np.ascontiguousarray(result).tobytes())
                    except (BrokenPipeError, ValueError) as exc:
                        raise MaxineError(
                            "编码进程提前退出，无法继续写入帧。\n"
                            + _log_tail(encode_log)
                        ) from exc

                    processed += 1
                    if preview_only and processed >= int(preview_frames):
                        break
                    if expected_total:
                        ratio = min(1.0, processed / expected_total)
                        emit(0.12 + ratio * 0.80,
                             f"Maxine 超分中 {processed}/{expected_total} 帧（{ratio * 100:.0f}%）")
                    elif processed % 5 == 0:
                        emit(0.5, f"Maxine 超分中 {processed} 帧…")
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
                raise MaxineError(
                    "没有任何一帧被成功解码，请确认视频未损坏。\n" + _log_tail(decode_log)
                )

            emit(0.95, "收尾编码…")
            try:
                encoder.stdin.close()
            except Exception:  # noqa: BLE001
                pass
            if encoder.wait() != 0 or not target.is_file():
                raise MaxineError("视频编码失败。\n" + _log_tail(encode_log))

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
            "pipeline": "stream",
        }
        emit(1.0, f"完成：{target.name}")
        logger.info(
            "Maxine 视频超分完成（流式）：%d 帧 / %.2fs -> %s（%.1f 帧/s）",
            processed, elapsed, target, processed / max(elapsed, 1e-6),
        )
        succeeded = True
        return target, result_info

    finally:
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
        if vsr is not None:
            vsr.release()
        if target is not None and target.exists() and not succeeded:
            target.unlink(missing_ok=True)
        if config.KEEP_TEMP_FILES:
            logger.info("保留临时目录用于排查：%s", work_dir)
        else:
            safe_remove(work_dir)


def _upscale_video_maxine_by_files(
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
    should_cancel: Callable[[], None] | None = None,
) -> tuple[Path, dict]:
    """用 Maxine VSR 对视频做时序超分（逐帧顺序推理）。

    与 RealESRGAN 链路一样，全程在 temp/ 子目录中操作并在结束时清理；
    ``should_cancel`` 会在每帧之前被调用，用于响应界面上的「取消任务」。
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
            if should_cancel is not None:
                should_cancel()
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
            "pipeline": "files",
        }
        emit(1.0, f"完成：{target.name}")
        logger.info("Maxine 视频超分完成：%d 帧 / %.2fs -> %s", processed, elapsed, target)
        return target, result_info

    finally:
        if vsr is not None:
            vsr.release()
        # 只清理本次任务自己的子目录。不要调用 cleanup_temp_dir()：它会清空
        # 整个 temp/，并发任务（Gradio 队列）之间会互相删掉对方的帧目录。
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
    should_cancel: Callable[[], None] | None = None,
) -> tuple[Path, dict]:
    """用 Maxine VSR 对视频做时序超分（逐帧顺序推理）。

    默认走**管道式管线**（解码 → VSR → 编码不落中间帧；Maxine 单帧约 10 ms，
    文件管线的 PNG 往返会成为主要开销）。异常时自动回退到文件管线。

    ``should_cancel`` 会在每帧之前被调用，用于响应界面上的「取消任务」。
    """
    _require_maxine()
    started = time.time()
    ffmpeg_utils.ensure_ffmpeg()
    device = config.normalize_device(device)

    source = Path(input_path)
    if not source.is_file():
        raise MaxineError(f"输入视频不存在：{source}")

    common = dict(
        output_path=output_path,
        preset_name=preset_name,
        device=device,
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

    if config.video_pipeline() == "stream":
        try:
            info = ffmpeg_utils.get_video_info(source)
        except ffmpeg_utils.FFmpegError as exc:
            raise MaxineError(f"无法读取视频信息，文件可能已损坏：{exc}") from exc
        try:
            return _maxine_streaming(source, info, started=started, **common)
        except MaxineError:
            raise
        except Exception as exc:  # noqa: BLE001 - 任何异常都退回文件管线
            logger.warning("Maxine 管道式管线不可用，已回退到文件管线：%s", exc)

    return _upscale_video_maxine_by_files(input_path, **common)


__all__ = [
    "HAS_MAXINE",
    "MaxineError",
    "MaxineVSR",
    "maxine_status",
    "upscale_video_maxine",
]
