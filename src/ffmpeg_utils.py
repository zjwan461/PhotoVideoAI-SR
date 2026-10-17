"""FFmpeg / FFprobe 封装：拆帧、提音频、合并视频、媒体信息探测。

全部通过 subprocess 调用，不依赖 ffmpeg-python 等第三方包。
所有失败都会抛出 :class:`FFmpegError`，并携带可读的 stderr 摘要。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path

import config
from src.common import get_logger

logger = get_logger("ffmpeg")

#: 拆帧输出的文件名前缀（固定宽度便于 ffmpeg 按序读取）
FRAME_PATTERN = "frame_%08d.png"
FRAME_GLOB = "frame_*.png"


class FFmpegError(RuntimeError):
    """FFmpeg/FFprobe 调用失败，message 已整理为用户可读文本。"""

    def __init__(self, message: str, stderr: str = "", command: Sequence[str] | None = None):
        super().__init__(message)
        self.stderr = stderr
        self.command = list(command) if command else []


# --------------------------------------------------------------------------- #
# 基础调用
# --------------------------------------------------------------------------- #
def _tail(text: str, lines: int = 12) -> str:
    """取 stderr 末尾若干行，避免把整段日志抛给用户。"""
    if not text:
        return ""
    chunks = [line for line in text.strip().splitlines() if line.strip()]
    return "\n".join(chunks[-lines:])


def run_ffmpeg(
    args: Sequence[str],
    *,
    timeout: int | None = None,
    check: bool = True,
    binary: str | None = None,
) -> subprocess.CompletedProcess:
    """执行一条 ffmpeg/ffprobe 命令。

    Args:
        args: 不含可执行文件本身的参数序列。
        timeout: 超时秒数，默认取 config.FFMPEG_TIMEOUT。
        check: 返回码非 0 时是否抛 :class:`FFmpegError`。
        binary: 覆盖可执行文件（用于 ffprobe）。

    Returns:
        ``subprocess.CompletedProcess``（text 模式）。
    """
    executable = binary or config.FFMPEG_BIN
    if shutil.which(executable) is None:
        raise FFmpegError(
            f"未找到可执行文件 `{executable}`。请安装 FFmpeg 并加入 PATH，"
            "或用环境变量 PVSR_FFMPEG / PVSR_FFPROBE 指定路径。",
            command=[executable, *args],
        )

    command = [executable, *[str(a) for a in args]]
    logger.debug("执行: %s", " ".join(command))
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout if timeout is not None else config.FFMPEG_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(
            f"FFmpeg 执行超时（超过 {timeout or config.FFMPEG_TIMEOUT} 秒），任务已中止。",
            command=command,
        ) from exc
    except OSError as exc:
        raise FFmpegError(f"无法启动 FFmpeg：{exc}", command=command) from exc

    if check and completed.returncode != 0:
        raise FFmpegError(
            f"FFmpeg 执行失败（退出码 {completed.returncode}）：\n{_tail(completed.stderr)}",
            stderr=completed.stderr,
            command=command,
        )
    return completed


def check_ffmpeg() -> str:
    """检测 FFmpeg 是否可用，返回可读状态文本（不抛异常）。"""
    return config.ffmpeg_diagnosis()


def ensure_ffmpeg() -> None:
    """FFmpeg 不可用时抛出 :class:`FFmpegError`。"""
    if not config.ffmpeg_available():
        raise FFmpegError(config.ffmpeg_diagnosis())


# --------------------------------------------------------------------------- #
# 版本探测与「帧率透传」参数
# --------------------------------------------------------------------------- #
_FFMPEG_VERSION: tuple[int, int] | None = None


def ffmpeg_version() -> tuple[int, int] | None:
    """探测 FFmpeg 版本，返回 ``(主, 次)``；探测失败返回 None（结果缓存）。"""
    global _FFMPEG_VERSION
    if _FFMPEG_VERSION is not None:
        return _FFMPEG_VERSION
    try:
        completed = run_ffmpeg(["-hide_banner", "-version"], check=False, timeout=30)
    except FFmpegError:
        return None
    text = (completed.stdout or "") + (completed.stderr or "")
    match = re.search(r"ffmpeg version\s+n?(\d+)\.(\d+)", text)
    if match:
        _FFMPEG_VERSION = (int(match.group(1)), int(match.group(2)))
    return _FFMPEG_VERSION


def passthrough_args(*, legacy: bool | None = None) -> list[str]:
    """返回「按源帧率透传、不做补帧/丢帧」的参数。

    FFmpeg 5.1 起用 ``-fps_mode passthrough`` 取代 ``-vsync 0``，而 ``-vsync``
    在较新版本中已被彻底移除（实测 FFmpeg 9.0 直接报
    ``Unrecognized option 'vsync'``）。这里按版本选择写法：

    * 主版本 >= 7：``-fps_mode passthrough``
    * 更早或版本未知：``-vsync 0``（调用方另有运行期回退）

    Args:
        legacy: 强制指定写法（True = ``-vsync``，False = ``-fps_mode``）；
            None 表示按探测到的版本自动选择。
    """
    if legacy is None:
        version = ffmpeg_version()
        legacy = bool(version and version[0] < 7)
    return ["-vsync", "0"] if legacy else ["-fps_mode", "passthrough"]


_NVENC_CACHE: dict[str, bool] = {}

#: NVENC 能力探测用的帧尺寸。必须明显大于 NVENC 的最小支持尺寸
#: （H.264 约 145x49），否则可用显卡会被误判为不可用。
_PROBE_WIDTH = 320
_PROBE_HEIGHT = 240


def reset_encoder_cache() -> None:
    """清空编码器能力判定缓存。

    换了 FFmpeg 可执行文件（例如刚下载了项目内自带版本）之后调用，
    否则会沿用旧二进制的判定结果。
    """
    _NVENC_CACHE.clear()


def has_nvenc(codec: str = "h264_nvenc") -> bool:
    """检测硬件编码器是否**真正可用**（结果缓存）。

    仅出现在 ``-encoders`` 列表里并不代表能跑：驱动缺失或运行时不匹配时，
    初始化会在实际编码时失败。这里做一次 1 帧的极小编码测试，
    不通过就让调用方回退到软件编码。

    注意：探测帧尺寸不能太小。NVENC 对输入尺寸有下限（H.264 约 145x49），
    用 64x64 之类的尺寸会让编码器以 ``Frame Dimension less than the minimum
    supported value`` 失败，从而把「完全可用的显卡」误判成不可用。
    因此这里用 320x240，并带上真实链路同样的 ``-pix_fmt yuv420p``。
    """
    if codec in _NVENC_CACHE:
        return _NVENC_CACHE[codec]

    def _listed() -> bool:
        try:
            completed = run_ffmpeg(["-hide_banner", "-encoders"], check=False, timeout=60)
            return codec in (completed.stdout or "")
        except FFmpegError:
            return False

    available = False
    if _listed():
        try:
            probe = run_ffmpeg(
                [
                    "-hide_banner", "-nostdin", "-v", "error",
                    "-f", "lavfi", "-i", f"color=c=black:s={_PROBE_WIDTH}x{_PROBE_HEIGHT}:d=0.04:r=25",
                    "-frames:v", "1", "-c:v", codec, "-pix_fmt", "yuv420p", "-f", "null", "-",
                ],
                check=False,
                timeout=120,
            )
            available = probe.returncode == 0
        except FFmpegError:
            available = False

    _NVENC_CACHE[codec] = available
    if not available:
        logger.warning("硬件编码器 %s 不可用（已列出=%s），将使用软件编码。", codec, _listed())
    return available


# --------------------------------------------------------------------------- #
# 媒体信息
# --------------------------------------------------------------------------- #
def _parse_fraction(value: str | None) -> float | None:
    """把 ``30000/1001`` 这样的分数转成 float。"""
    if not value:
        return None
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            denominator_f = float(denominator)
            if denominator_f == 0:
                return None
            return float(numerator) / denominator_f
        return float(value)
    except (TypeError, ValueError):
        return None


def get_video_info(path: str | Path) -> dict:
    """用 ffprobe 读取视频信息。

    Returns:
        dict，含 width / height / fps / duration / nb_frames / has_audio /
        video_codec / audio_codec / size_bytes；字段不可得时为 None。
    """
    ensure_ffmpeg()
    source = Path(path)
    if not source.is_file():
        raise FFmpegError(f"输入视频不存在：{source}")

    args = [
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(source),
    ]
    completed = run_ffmpeg(args, binary=config.FFPROBE_BIN, check=False)
    if completed.returncode != 0:
        raise FFmpegError(
            f"无法解析视频文件（文件可能已损坏或格式不受支持）：{source.name}\n"
            f"{_tail(completed.stderr, 6)}",
            stderr=completed.stderr,
            command=[config.FFPROBE_BIN, *args],
        )
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"无法解析 ffprobe 输出：{exc}", stderr=completed.stdout) from exc

    video_stream = None
    audio_stream = None
    for stream in payload.get("streams", []):
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        elif stream.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = stream

    if video_stream is None:
        raise FFmpegError("输入文件中没有可用的视频流，请确认文件未损坏。")

    fmt = payload.get("format", {})
    fps = (
        _parse_fraction(video_stream.get("avg_frame_rate"))
        or _parse_fraction(video_stream.get("r_frame_rate"))
        or config.DEFAULT_FPS
    )
    if fps <= 0:
        fps = config.DEFAULT_FPS

    duration = None
    for raw in (fmt.get("duration"), video_stream.get("duration")):
        try:
            if raw is not None:
                duration = float(raw)
                break
        except (TypeError, ValueError):
            continue

    nb_frames = None
    try:
        if video_stream.get("nb_frames") is not None:
            nb_frames = int(video_stream["nb_frames"])
    except (TypeError, ValueError):
        nb_frames = None
    if nb_frames is None and duration:
        nb_frames = int(round(duration * fps))

    info = {
        "path": str(source),
        "width": int(video_stream.get("width") or 0) or None,
        "height": int(video_stream.get("height") or 0) or None,
        "fps": round(float(fps), 4),
        "duration": round(duration, 3) if duration else None,
        "nb_frames": nb_frames,
        "has_audio": audio_stream is not None,
        "video_codec": video_stream.get("codec_name"),
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
        "size_bytes": int(fmt.get("size") or source.stat().st_size),
    }
    return info


def format_video_info(info: dict) -> str:
    """把 get_video_info 的结果格式化成 UI 文案。"""
    if not info:
        return "无媒体信息"
    duration = f"{info['duration']:.2f}s" if info.get("duration") else "未知"
    frames = info.get("nb_frames") if info.get("nb_frames") else "未知"
    return (
        f"分辨率: {info.get('width')}x{info.get('height')} | "
        f"帧率: {info.get('fps')} fps | 时长: {duration} | 帧数: {frames} | "
        f"视频编码: {info.get('video_codec')} | "
        f"音频: {'有 (' + str(info.get('audio_codec')) + ')' if info.get('has_audio') else '无'}"
    )


# --------------------------------------------------------------------------- #
# 拆帧 / 提音频
# --------------------------------------------------------------------------- #
def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    fps: float | None = None,
    max_frames: int | None = None,
    start_time: float | None = None,
    image_format: str = "PNG",
) -> list[Path]:
    """把视频拆成图片帧。

    Args:
        video_path: 源视频。
        output_dir: 帧输出目录（会自动创建）。
        fps: 目标采样帧率；None 表示保持源帧率。
        max_frames: 最多输出多少帧（预览模式用）。
        start_time: 从第几秒开始。
        image_format: ``PNG`` / ``JPG``。

    Returns:
        按文件名排序的帧路径列表。
    """
    ensure_ffmpeg()
    source = Path(video_path)
    if not source.is_file():
        raise FFmpegError(f"输入视频不存在：{source}")

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    extension = "jpg" if image_format.upper() == "JPG" else "png"
    pattern = f"frame_%08d.{extension}"
    base_args: list[str] = ["-hide_banner", "-nostdin", "-y"]
    if start_time:
        base_args += ["-ss", f"{float(start_time):.3f}"]
    base_args += ["-i", str(source)]
    if fps:
        base_args += ["-vf", f"fps={float(fps):.6f}"]
    if max_frames:
        base_args += ["-frames:v", str(int(max_frames))]
    if extension == "jpg":
        base_args += ["-q:v", "2"]
    output = str(target_dir / pattern)

    def _build(mode_args: Sequence[str]) -> list[str]:
        return [*base_args, *mode_args, output]

    preferred = passthrough_args()
    try:
        run_ffmpeg(_build(preferred))
    except FFmpegError as exc:
        # 版本探测可能不准（自定义构建 / 包装脚本），这里做一次运行期回退：
        # 只要报的是「不认识这个选项」，就换另一种帧率透传写法重试。
        if "Unrecognized option" not in str(exc):
            raise
        fallback = passthrough_args(legacy=(preferred[0] == "-fps_mode"))
        logger.warning(
            "当前 FFmpeg 不支持 %s，改用 %s 重试拆帧。", preferred[0], fallback[0]
        )
        run_ffmpeg(_build(fallback))

    frames = sorted(target_dir.glob(f"frame_*.{extension}"))
    if not frames:
        raise FFmpegError("拆帧结果为空，请确认视频未损坏且包含有效画面。")
    logger.info("拆帧完成：%d 帧 -> %s", len(frames), target_dir)
    return frames


def preview_extract(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    frame_limit: int | None = None,
    image_format: str = "PNG",
) -> list[Path]:
    """只提取前 N 帧，用于「仅预览」模式（P1）。"""
    limit = int(frame_limit or config.PREVIEW_FRAME_LIMIT)
    return extract_frames(
        video_path, output_dir, max_frames=limit, image_format=image_format
    )


def extract_audio(
    video_path: str | Path,
    output_path: str | Path,
    *,
    audio_codec: str = "aac",
    bitrate: str = "192k",
) -> Path | None:
    """提取音轨；源视频没有音频时返回 None（不算失败）。"""
    ensure_ffmpeg()
    source = Path(video_path)
    if not source.is_file():
        raise FFmpegError(f"输入视频不存在：{source}")

    try:
        info = get_video_info(source)
    except FFmpegError:
        return None
    if not info.get("has_audio"):
        logger.info("源视频无音轨，跳过音频提取。")
        return None

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "-hide_banner", "-nostdin", "-y",
        "-i", str(source),
        "-vn",
    ]
    if audio_codec == "copy":
        args += ["-acodec", "copy"]
    else:
        args += ["-acodec", audio_codec, "-b:a", bitrate]
    args += [str(target)]
    run_ffmpeg(args)
    logger.info("音频提取完成 -> %s", target)
    return target if target.exists() else None


# --------------------------------------------------------------------------- #
# 管道式解码 / 编码（不落中间帧文件）
# --------------------------------------------------------------------------- #
#: 管道里统一的像素格式：RGB 与项目内部数组一致，省掉两次色序转换
PIPE_PIX_FMT = "rgb24"


def read_exact(stream, size: int) -> bytes | None:
    """从管道里精确读 ``size`` 字节。

    Returns:
        读满时返回 bytes；在**帧边界**遇到 EOF（视频正常结束）返回 None；
        读到一半就 EOF 表示解码异常，抛 :class:`FFmpegError`。
    """
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            if remaining == size:
                return None
            raise FFmpegError(
                f"解码管道提前结束：本帧还差 {remaining} 字节，视频可能已损坏。"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def spawn_frame_decoder(source: str | Path, *, stderr=None, pix_fmt: str = PIPE_PIX_FMT):
    """启动解码进程：把视频逐帧以 rawvideo 写到 ``stdout``。

    Args:
        source: 输入视频。
        stderr: 重定向目标（文件对象）；为 None 时丢弃，避免管道写满导致死锁。
        pix_fmt: 输出像素格式，默认 ``rgb24``。

    Returns:
        ``subprocess.Popen``；调用方负责读 ``stdout`` 并 ``terminate``/``wait``。
    """
    ensure_ffmpeg()
    source_path = Path(source)
    if not source_path.is_file():
        raise FFmpegError(f"输入视频不存在：{source_path}")
    command = [
        config.FFMPEG_BIN, "-hide_banner", "-nostdin", "-v", "error",
        "-i", str(source_path),
        "-map", "0:v:0",
        "-f", "rawvideo", "-pix_fmt", pix_fmt, "-",
    ]
    logger.debug("解码进程：%s", " ".join(command))
    try:
        return subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=stderr if stderr is not None else subprocess.DEVNULL,
            bufsize=0,
        )
    except OSError as exc:
        raise FFmpegError(f"无法启动解码进程：{exc}", command=command) from exc


def even_size(value: int) -> int:
    """把尺寸向下取整到偶数（yuv420p 要求宽高为偶数）。"""
    value = int(value)
    return value if value % 2 == 0 else value - 1


def spawn_frame_encoder(
    output_path: str | Path,
    width: int,
    height: int,
    *,
    fps: float,
    stderr=None,
    audio_path: str | Path | None = None,
    video_format: str = "MP4",
    codec: str = "h264_nvenc",
    bitrate_mbps: float | None = None,
    crf: int | None = None,
    pix_fmt: str = "yuv420p",
):
    """启动编码进程：从 ``stdin`` 读 rawvideo 帧，编码成视频文件。

    参数语义与 :func:`combine_frames_to_video` 保持一致（质量模式 / 容器 / 音轨）。

    Returns:
        ``subprocess.Popen``；调用方写完 ``stdin`` 后要 ``close()`` 再 ``wait()``。
    """
    ensure_ffmpeg()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    video_format = video_format.upper()
    container = config.VIDEO_CODEC_MAP.get(video_format, config.VIDEO_CODEC_MAP["MP4"])

    out_w, out_h = even_size(width), even_size(height)
    if (out_w, out_h) != (int(width), int(height)):
        logger.warning(
            "输出尺寸 %dx%d 不是偶数，已调整为 %dx%d（yuv420p 要求）",
            width, height, out_w, out_h,
        )
    has_audio = bool(audio_path) and Path(audio_path).is_file()

    command = [
        config.FFMPEG_BIN, "-hide_banner", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", PIPE_PIX_FMT,
        "-s", f"{out_w}x{out_h}",
        "-framerate", f"{float(fps):.6f}",
        "-i", "-",
    ]
    if has_audio:
        command += ["-i", str(audio_path)]

    selected = codec if (codec == "libx264" or has_nvenc(codec)) else "libx264"
    command += ["-c:v", selected]
    if selected == "h264_nvenc":
        if bitrate_mbps:
            command += [
                "-b:v", f"{int(float(bitrate_mbps) * 1_000_000)}",
                "-maxrate", f"{int(float(bitrate_mbps) * 1.5 * 1_000_000)}",
            ]
        else:
            command += ["-rc", "vbr", "-cq", str(int(crf if crf is not None else config.FFMPEG_CRF)), "-b:v", "0"]
        command += ["-preset", "p5", "-tune", "hq", "-pix_fmt", pix_fmt]
    else:
        command += ["-preset", config.FFMPEG_X264_PRESET, "-pix_fmt", pix_fmt]
        if bitrate_mbps:
            command += ["-b:v", f"{int(float(bitrate_mbps) * 1_000_000)}"]
        else:
            command += ["-crf", str(int(crf if crf is not None else config.FFMPEG_CRF))]
    if has_audio:
        command += ["-c:a", container["audio_codec"], "-shortest"]
    if video_format == "MP4":
        command += ["-movflags", "+faststart"]
    command += [str(target)]

    logger.debug("编码进程：%s", " ".join(command))
    try:
        return subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=stderr if stderr is not None else subprocess.DEVNULL,
            bufsize=0,
        )
    except OSError as exc:
        raise FFmpegError(f"无法启动编码进程：{exc}", command=command) from exc


# --------------------------------------------------------------------------- #
# 合并
# --------------------------------------------------------------------------- #
def combine_frames_to_video(
    frames_dir: str | Path,
    output_path: str | Path,
    *,
    fps: float = 25.0,
    audio_path: str | Path | None = None,
    image_format: str = "PNG",
    video_format: str = "MP4",
    codec: str = "h264_nvenc",
    bitrate_mbps: float | None = None,
    crf: int | None = None,
    pix_fmt: str = "yuv420p",
    allow_software_fallback: bool = True,
) -> Path:
    """把图片序列（+可选音轨）合成为视频。

    优先使用硬件编码（``h264_nvenc``）；不可用或失败时按需回退 ``libx264``。
    """
    ensure_ffmpeg()
    source_dir = Path(frames_dir)
    extension = "jpg" if image_format.upper() == "JPG" else "png"
    sample = next(iter(sorted(source_dir.glob(f"frame_*.{extension}"))), None)
    if sample is None:
        raise FFmpegError(f"目录中没有可合成的帧：{source_dir}")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    video_format = video_format.upper()
    container = config.VIDEO_CODEC_MAP.get(video_format, config.VIDEO_CODEC_MAP["MP4"])

    def build_args(selected_codec: str) -> list[str]:
        args = [
            "-hide_banner", "-nostdin", "-y",
            "-framerate", f"{float(fps):.6f}",
            "-i", str(source_dir / f"frame_%08d.{extension}"),
        ]
        if audio_path and Path(audio_path).is_file():
            args += ["-i", str(audio_path)]
        args += ["-c:v", selected_codec]
        if selected_codec == "h264_nvenc":
            if bitrate_mbps:
                args += ["-b:v", f"{int(float(bitrate_mbps) * 1_000_000)}", "-maxrate", f"{int(float(bitrate_mbps) * 1.5 * 1_000_000)}"]
            else:
                # nvenc 的 -cq 近似 CRF，配合 -rc vbr 可获得稳定质量
                args += ["-rc", "vbr", "-cq", str(int(crf if crf is not None else config.FFMPEG_CRF)), "-b:v", "0"]
            args += ["-preset", "p5", "-tune", "hq", "-pix_fmt", pix_fmt]
        else:
            args += ["-preset", config.FFMPEG_X264_PRESET, "-pix_fmt", pix_fmt]
            if bitrate_mbps:
                args += ["-b:v", f"{int(float(bitrate_mbps) * 1_000_000)}"]
            else:
                args += ["-crf", str(int(crf if crf is not None else config.FFMPEG_CRF))]
        if audio_path and Path(audio_path).is_file():
            args += ["-c:a", container["audio_codec"], "-shortest"]
        args += ["-movflags", "+faststart"] if video_format == "MP4" else []
        args += [str(target)]
        return args

    selected = codec if (codec == "libx264" or has_nvenc(codec)) else "libx264"
    try:
        run_ffmpeg(build_args(selected))
    except FFmpegError as exc:
        if selected != "libx264" and allow_software_fallback:
            logger.warning("硬件编码失败，回退软件编码：%s", _tail(str(exc), 3))
            run_ffmpeg(build_args("libx264"))
        else:
            raise

    if not target.exists() or target.stat().st_size == 0:
        raise FFmpegError("视频合成失败：输出文件为空。")
    logger.info("视频合成完成 -> %s", target)
    return target


def mux_audio(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    *,
    video_format: str = "MP4",
) -> Path:
    """给已有视频补上音轨（不重新编码视频）。"""
    ensure_ffmpeg()
    container = config.VIDEO_CODEC_MAP.get(video_format.upper(), config.VIDEO_CODEC_MAP["MP4"])
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "-hide_banner", "-nostdin", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", container["audio_codec"], "-shortest",
        str(target),
    ]
    run_ffmpeg(args)
    return target


def concat_videos(
    video_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    video_format: str = "MP4",
) -> Path:
    """用 concat demuxer 拼接多个编码参数一致的视频。"""
    ensure_ffmpeg()
    paths = [Path(p) for p in video_paths]
    if not paths:
        raise FFmpegError("拼接列表为空。")
    list_file = Path(output_path).parent / "concat_list.txt"
    list_file.parent.mkdir(parents=True, exist_ok=True)
    list_file.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in paths), encoding="utf-8"
    )
    target = Path(output_path)
    args = [
        "-hide_banner", "-nostdin", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(target),
    ]
    run_ffmpeg(args)
    return target


def probe_encoder_support() -> str:
    """返回编码器可用性说明（供 UI 展示）。"""
    lines = []
    lines.append("h264_nvenc: " + ("✅ 可用" if has_nvenc("h264_nvenc") else "❌ 不可用（将回退 libx264）"))
    lines.append("hevc_nvenc: " + ("✅ 可用" if has_nvenc("hevc_nvenc") else "❌ 不可用"))
    return "\n".join(lines)


__all__ = [
    "FFmpegError",
    "run_ffmpeg",
    "check_ffmpeg",
    "ensure_ffmpeg",
    "has_nvenc",
    "get_video_info",
    "format_video_info",
    "extract_frames",
    "preview_extract",
    "extract_audio",
    "combine_frames_to_video",
    "mux_audio",
    "concat_videos",
    "probe_encoder_support",
    "reset_encoder_cache",
    "ffmpeg_version",
    "passthrough_args",
    "FRAME_PATTERN",
    "FRAME_GLOB",
]
