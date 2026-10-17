"""FFmpeg 自动下载：给没有预装 FFmpeg 的用户一键装到项目内 ``tools/ffmpeg/``。

为什么需要
----------
本项目用 FFmpeg 完成全部媒体编解码。Windows 不自带 FFmpeg，部分 Linux 环境也没有；
缺了它视频链路完全不可用，而让用户自己去官网找压缩包、解压、配 PATH 是最大的劝退点。
这里提供「下载即用」的兜底：

* 下载主流预编译包，只把 ``ffmpeg`` / ``ffprobe`` 两个可执行文件放进
  ``tools/ffmpeg/``，**不污染系统 PATH、不需要管理员权限**；
* 装好后 :func:`config.refresh_ffmpeg_bins` 会让 ``config.FFMPEG_BIN`` 指向它，
  全部任务立即生效（优先级：环境变量 > 项目内自带 > PATH）；
* 只支持 Windows / Linux —— 本项目仅支持 NVIDIA CUDA 推理，macOS 天然不在范围内，
  其它平台会给出明确的手工安装指引。

安全与健壮性
------------
* 支持 ``PVSR_FFMPEG_URL`` 覆盖下载地址，方便国内网络、内网镜像或代理下载；
* 下载源按顺序回退，单个源失败不会让整个流程失败；
* 解压前逐个校验成员路径，拒绝 ``..`` 越界写入与符号链接；
* 安装后立刻执行一次 ``ffmpeg -version`` 自检，不把坏包留在磁盘上。
"""

from __future__ import annotations

import os
import platform
import shutil
import stat
import sys
import tarfile
import time
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import config
from src.common import get_logger, human_size

logger = get_logger("ffmpeg_install")

#: 部分 CDN 会拒绝 urllib 的默认 UA
_USER_AGENT = (
    "PhotoVideoAI-SR/1.0 (+https://github.com/zjwan461/PhotoVideoAI-SR)"
)

#: Windows：优先 GitHub 的 BtbN 构建（走 CDN，实测比 gyan.dev 快两个数量级）；
#: gyan.dev 作为备用主机（部分内网只放行它）。两者都含 ffmpeg.exe / ffprobe.exe。
_WINDOWS_URLS: tuple[str, ...] = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
    "ffmpeg-master-latest-win64-gpl.zip",
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
)
#: Linux：johnvansickle 的静态构建，单文件无依赖
_LINUX_AMD64_URLS: tuple[str, ...] = (
    "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
)
_LINUX_ARM64_URLS: tuple[str, ...] = (
    "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz",
)

#: 低速放弃阈值：连续 ``_SLOW_WINDOW`` 秒内平均速度低于 ``_SLOW_LIMIT`` 就换下一个源。
#: 用来对付「不报错、但是每秒几 KB」的源（否则一个坏源可能卡住几小时）。
_SLOW_WINDOW = 20.0
_SLOW_LIMIT = 64 * 1024


class FFmpegInstallError(RuntimeError):
    """FFmpeg 下载/解压失败，message 已是面向用户的提示。"""


# --------------------------------------------------------------------------- #
# 平台与下载地址
# --------------------------------------------------------------------------- #
def platform_key() -> str:
    """返回目标平台标识：``windows`` / ``linux`` / ``unsupported``。"""
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform.startswith("linux"):
        return "linux"
    return "unsupported"


def linux_arch() -> str:
    """返回 Linux 架构标识：``amd64`` / ``arm64``。"""
    machine = platform.machine().lower()
    return "arm64" if machine in {"aarch64", "arm64"} else "amd64"


def is_supported() -> bool:
    """当前平台是否支持自动下载。"""
    return platform_key() != "unsupported"


def candidate_urls() -> list[str]:
    """返回本次可尝试的下载地址（按优先级）。

    环境变量 ``PVSR_FFMPEG_URL`` 可用于内网镜像、代理下载或指定其它版本。
    """
    override = os.environ.get("PVSR_FFMPEG_URL", "").strip()
    if override:
        return [override]
    key = platform_key()
    if key == "windows":
        return list(_WINDOWS_URLS)
    if key == "linux":
        return list(_LINUX_ARM64_URLS if linux_arch() == "arm64" else _LINUX_AMD64_URLS)
    return []


def manual_hint() -> str:
    """不支持的平台上给用户的手工安装指引。"""
    return (
        "当前系统不支持自动下载 FFmpeg（本项目面向 Windows / Linux）。\n"
        "请手动安装后重试：\n"
        "  * Ubuntu / Debian：sudo apt install ffmpeg\n"
        "  * macOS：brew install ffmpeg\n"
        "然后用环境变量 PVSR_FFMPEG / PVSR_FFPROBE 指定可执行文件路径（或加入 PATH）。"
    )


# --------------------------------------------------------------------------- #
# 下载 / 解压
# --------------------------------------------------------------------------- #
def _emit(progress: Callable[[float, str], None] | None, value: float, message: str) -> None:
    """安全地汇报进度：回调异常不应中断安装。"""
    if progress is None:
        return
    try:
        progress(max(0.0, min(1.0, value)), message)
    except Exception:  # noqa: BLE001 - 进度回调失败与安装结果无关
        logger.debug("进度回调异常（已忽略）：%s", message)


def _download(
    url: str,
    target: Path,
    *,
    progress: Callable[[float, str], None] | None = None,
    timeout: int = 45,
) -> None:
    """下载到 ``target``；持续低速时主动放弃，让调用方换下一个源。

    ``timeout`` 是**单次读**的超时（``urlopen`` 的 socket 超时），
    稳定但慢的连接不会被它打断；真正的「慢到不可用」由 ``_SLOW_*`` 判定。
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        total = int(response.headers.get("Content-Length") or 0)
        received = 0
        with open(target, "wb") as handle:
            while True:
                chunk = response.read(1024 * 512)
                if not chunk:
                    break
                handle.write(chunk)
                received += len(chunk)
                elapsed = time.time() - started
                if total:
                    _emit(
                        progress,
                        0.05 + 0.75 * received / total,
                        f"下载 FFmpeg：{human_size(received)} / {human_size(total)}",
                    )
                if elapsed > _SLOW_WINDOW and received / elapsed < _SLOW_LIMIT:
                    raise FFmpegInstallError(
                        f"下载速度过低（约 {received / elapsed / 1024:.0f} KB/s），"
                        "已放弃该下载源。"
                    )
    if not target.is_file() or target.stat().st_size == 0:
        raise FFmpegInstallError("下载到的文件为空，可能是网络中断或地址失效。")


def _safe_target(root: Path, name: str) -> Path:
    """校验压缩包成员不会写到 ``root`` 之外。"""
    root_resolved = root.resolve()
    target = (root_resolved / name).resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise FFmpegInstallError(f"压缩包内含越界路径，已中止：{name}")
    return target


def _extract(archive: Path, dest: Path) -> None:
    """把 zip / tar.xz 安全解压到 ``dest``。"""
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                _safe_target(dest, member)
            bundle.extractall(dest)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                _safe_target(dest, member.name)
                if member.issym() or member.islnk():
                    raise FFmpegInstallError(f"压缩包内含符号链接，已中止：{member.name}")
            try:  # Python 3.12+ 默认收紧解压行为，显式声明更安全
                bundle.extractall(dest, filter="data")
            except TypeError:  # 老版本没有 filter 参数
                bundle.extractall(dest)
        return
    raise FFmpegInstallError(
        "下载到的文件既不是 zip 也不是 tar 包，可能地址被重定向到了错误页面。"
    )


def _binary_names() -> tuple[str, str]:
    """返回 ``(ffmpeg 文件名, ffprobe 文件名)``（Windows 带 .exe）。"""
    suffix = ".exe" if platform_key() == "windows" else ""
    return f"ffmpeg{suffix}", f"ffprobe{suffix}"


def _locate(root: Path, filename: str) -> Path:
    """在解压目录里找可执行文件，优先 ``bin/`` 下的那份。"""
    candidates = [path for path in root.rglob(filename) if path.is_file()]
    if not candidates:
        raise FFmpegInstallError(
            f"压缩包里没有找到 {filename}，可能下载到了错误的文件。"
        )
    candidates.sort(key=lambda path: (path.parent.name != "bin", len(path.parts)))
    return candidates[0]


# --------------------------------------------------------------------------- #
# 安装
# --------------------------------------------------------------------------- #
def verify() -> str:
    """执行一次 ``ffmpeg -version`` 自检，返回版本首行。"""
    from src import ffmpeg_utils

    try:
        completed = ffmpeg_utils.run_ffmpeg(
            ["-hide_banner", "-version"], check=False, timeout=60
        )
    except ffmpeg_utils.FFmpegError as exc:
        raise FFmpegInstallError(f"FFmpeg 安装后仍无法执行：{exc}") from exc
    if completed.returncode != 0:
        raise FFmpegInstallError("FFmpeg 安装后无法正常执行（退出码非 0）。")
    if not config.ffmpeg_available():
        raise FFmpegInstallError(
            "FFmpeg 已下载，但 ffprobe 不可用；请检查 "
            f"{config.FFMPEG_BUNDLE_DIR} 目录是否完整。"
        )
    lines = (completed.stdout or completed.stderr or "").strip().splitlines()
    return lines[0] if lines else "ffmpeg"


def install(
    *,
    progress: Callable[[float, str], None] | None = None,
    timeout: int = 45,
) -> Path:
    """下载并安装 FFmpeg 到项目内，返回可执行文件路径。

    已经可用时直接返回现有路径（幂等）。

    Args:
        progress: 进度回调 ``(value, message)``，可为 None。
        timeout: 单次网络读取的超时（秒）。

    Raises:
        FFmpegInstallError: 平台不支持、所有下载源都失败、或安装后自检不通过。
    """
    if not is_supported():
        raise FFmpegInstallError(manual_hint())

    if config.ffmpeg_available():
        # 已经能用（系统 PATH 或环境变量指定）：解析成真实路径再返回，直接短路
        existing = shutil.which(config.FFMPEG_BIN) or config.FFMPEG_BIN
        logger.info("FFmpeg 已可用，无需下载：%s", existing)
        _emit(progress, 1.0, "FFmpeg 已就绪")
        return Path(existing)

    urls = candidate_urls()
    if not urls:
        raise FFmpegInstallError(manual_hint())

    work_dir = config.TEMP_DIR / f"ffmpeg_install_{os.getpid()}"
    archive: Path | None = None
    failures: list[str] = []
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        _emit(progress, 0.02, "准备下载 FFmpeg…")

        for index, url in enumerate(urls, start=1):
            filename = Path(urlparse(url).path).name or "ffmpeg-download"
            target = work_dir / filename
            try:
                logger.info("下载 FFmpeg（源 %d/%d）：%s", index, len(urls), url)
                _download(url, target, progress=progress, timeout=timeout)
                archive = target
                break
            except FFmpegInstallError as exc:
                failures.append(f"{url}\n    -> {exc}")
                target.unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001 - 网络异常类型很多
                failures.append(f"{url}\n    -> {exc}")
                target.unlink(missing_ok=True)
                logger.warning("下载源失败，尝试下一个：%s（%s）", url, exc)

        if archive is None:
            raise FFmpegInstallError(
                "FFmpeg 下载失败，已尝试以下地址：\n\n"
                + "\n".join(f"  * {item}" for item in failures)
                + "\n\n可用环境变量 PVSR_FFMPEG_URL 指定其它下载地址（内网镜像 / 代理），\n"
                "或手动安装 FFmpeg 并加入 PATH（也可用 PVSR_FFMPEG 指定绝对路径）。"
            )

        _emit(progress, 0.82, "解压 FFmpeg…")
        staging = work_dir / "unpacked"
        _extract(archive, staging)

        _emit(progress, 0.90, "安装到项目内…")
        names = _binary_names()
        config.FFMPEG_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
        installed: dict[str, Path] = {}
        for filename in names:
            source = _locate(staging, filename)
            destination = config.FFMPEG_BUNDLE_DIR / filename
            shutil.copy2(source, destination)
            if os.name != "nt":
                mode = destination.stat().st_mode
                destination.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            installed[filename] = destination
            logger.info("已安装 %s -> %s", filename, destination)

        # 让 config.FFMPEG_BIN / FFPROBE_BIN 立即指向新文件，并清掉编码器判定缓存
        config.refresh_ffmpeg_bins()
        from src import ffmpeg_utils

        ffmpeg_utils.reset_encoder_cache()

        _emit(progress, 0.96, "验证 FFmpeg 可用性…")
        version = verify()
        logger.info("FFmpeg 安装完成：%s", version)
        _emit(progress, 1.0, "FFmpeg 已就绪")
        return installed[names[0]]
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def status_text() -> str:
    """返回一行可读状态，供 UI 展示。"""
    if config.ffmpeg_available():
        return f"✅ FFmpeg 可用：`{config.FFMPEG_BIN}`"
    if not is_supported():
        return "⚠️ 当前系统不支持自动下载，请手动安装 FFmpeg。"
    return (
        f"❌ 未找到 FFmpeg。点上方「⬇️ 下载 FFmpeg」可自动安装到 "
        f"`{config.FFMPEG_BUNDLE_DIR}`（约 100 MB）。"
    )


__all__ = [
    "FFmpegInstallError",
    "candidate_urls",
    "install",
    "is_supported",
    "linux_arch",
    "manual_hint",
    "platform_key",
    "status_text",
    "verify",
]
