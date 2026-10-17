#!/usr/bin/env python3
"""PhotoVideoAI-SR 基础自测脚本（P0 自测清单的可执行版本）。

特点：
* **不需要 GPU**：媒体链路与工具函数部分在纯 CPU 环境即可运行；
* 有 CUDA 且 `model_weights/` 中存在基础模型时，会额外跑一遍真实图片超分；
* 不依赖 pytest，直接 `python tests/selftest.py` 即可。

退出码 0 表示全部通过，1 表示有失败项。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import ffmpeg_utils  # noqa: E402
from src.common import (  # noqa: E402
    lora_choices,
    make_temp_subdir,
    resolve_lora_path,
    safe_remove,
    scan_lora_weights,
)

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  ✅ {name}")
    else:
        FAILED.append(f"{name} {detail}".strip())
        print(f"  ❌ {name} {detail}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# --------------------------------------------------------------------------- #
def test_config_and_common() -> None:
    section("配置与通用工具")
    check("所有目录已创建", all(p.exists() for p in config.REQUIRED_DIRS))
    check("分辨率预设可用", {"1080p (1920x1080)", "4K (3840x2160)"} <= set(config.RESOLUTION_PRESETS))
    check("设备下拉可渲染", isinstance(config.get_gpu_options(), list) and len(config.get_gpu_options()) >= 1)
    check("设备字符串解析", config.parse_gpu_option("cuda:1 | anything") == "cuda:1" and config.parse_gpu_option(None) == "cuda:0")
    check("LoRA 下拉含占位项", lora_choices()[0].startswith("不使用"))
    check("越权 LoRA 路径被拒绝", resolve_lora_path("/etc/passwd") is None)
    check("扫描 LoRA 目录不报错", isinstance(scan_lora_weights(), list))


def test_ffmpeg_chain() -> None:
    section("FFmpeg 媒体链路")
    if not config.ffmpeg_available():
        check("FFmpeg 可用", False, "未安装 FFmpeg，跳过媒体链路测试")
        return
    check("FFmpeg 状态文本", ffmpeg_utils.check_ffmpeg().startswith("✅"))

    work = make_temp_subdir("selftest")
    source = work / "source.mp4"
    create = subprocess.run(
        [
            config.FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    if create.returncode != 0:
        check("生成测试视频", False, create.stderr[-200:])
        cleanup_temp_dir()
        return
    check("生成测试视频", True)

    try:
        info = ffmpeg_utils.get_video_info(source)
        check("探测视频信息", info["width"] == 160 and info["height"] == 120 and info["has_audio"])
        frames = ffmpeg_utils.extract_frames(source, work / "frames")
        check("拆帧", len(frames) == 10, f"得到 {len(frames)} 帧")
        preview = ffmpeg_utils.preview_extract(source, work / "preview", frame_limit=3)
        check("预览拆帧", len(preview) == 3, f"得到 {len(preview)} 帧")
        audio = ffmpeg_utils.extract_audio(source, work / "audio.m4a")
        check("提取音轨", audio is not None and audio.exists())
        out = ffmpeg_utils.combine_frames_to_video(
            work / "frames", work / "out.mp4", fps=info["fps"], audio_path=audio
        )
        out_info = ffmpeg_utils.get_video_info(out)
        check("合成视频并保留音轨", out_info["has_audio"] and out_info["nb_frames"] == 10)
        check("编码器探测可返回结果", isinstance(ffmpeg_utils.has_nvenc(), bool))

        # 损坏输入必须给出可读错误
        broken = work / "broken.mp4"
        broken.write_bytes(b"this is not a video")
        try:
            ffmpeg_utils.get_video_info(broken)
            check("损坏视频报错", False, "未抛出异常")
        except ffmpeg_utils.FFmpegError as exc:
            check("损坏视频报错", "损坏" in str(exc), str(exc)[:60])
    finally:
        # 只清理本次测试自己的子目录：temp/ 里可能有其它实例（如正在跑的 WebUI）
        safe_remove(work)

    check("临时目录已清空", not work.exists())


def test_cuda_and_inference() -> None:
    section("CUDA 与推理（可选）")
    use_cpu = os.environ.get("PVSR_SELFTEST_CPU", "0") in ("1", "true", "True")
    if not config.cuda_available() and not use_cpu:
        print("  ⏭  未检测到可用 CUDA，跳过推理测试（这不代表代码有问题）")
        print("     想用 CPU 慢速验证逻辑，可设置 PVSR_SELFTEST_CPU=1 再运行")
        return

    from src import img_sr

    if not img_sr.HAS_REALESRGAN:
        print("  ⏭  未安装 realesrgan/basicsr，跳过推理测试")
        return

    weights = img_sr.model_path(config.DEFAULT_MODEL_NAME)
    if not weights.is_file():
        print(f"  ⏭  缺少模型权重 {weights.name}，跳过推理测试")
        return

    device = config.get_gpu_options()[0] if config.cuda_available() else "cpu"

    try:
        import numpy as np

        sample = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype("uint8")
        result, info = img_sr.upscale_image(
            sample, preset_name="原始比例 (不限制长边)", device=device, auto_download=False
        )
        check("图片超分推理", result.shape[0] == 256 and result.shape[1] == 256, str(result.shape))
        print(f"     设备 {device}，耗时 {info['elapsed']:.2f}s，{info['input_size']} -> {info['output_size']}")
    except Exception as exc:  # noqa: BLE001
        check("图片超分推理", False, str(exc)[:120])


def main() -> int:
    print("PhotoVideoAI-SR 自测开始")
    print(config.runtime_summary())
    test_config_and_common()
    test_ffmpeg_chain()
    test_cuda_and_inference()

    print("\n" + "=" * 60)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    for item in FAILED:
        print(f"  - 失败：{item}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
