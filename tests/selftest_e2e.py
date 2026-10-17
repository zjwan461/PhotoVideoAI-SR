#!/usr/bin/env python3
"""端到端自测（需要 CUDA + 基础模型权重）：验证「真机可用」而不只是「能导入」。

与另外两个自测的区别：

* ``tests/selftest.py``      —— 配置 / 媒体链路 / 单张推理，权重缺失时跳过推理；
* ``tests/selftest_train.py`` —— 微调模块，CPU + 微型网络即可跑完；
* 本脚本                    —— **全链路真机验收**：真实权重、真实 GPU、真实视频，
  覆盖 UI 处理器与事件接线，能抓出「导入没问题、一点按钮就崩」的缺陷。

覆盖内容
--------
1. 图片超分：分辨率预设（4K）、FP16、UI 下拉标签作为设备参数的归一化
2. LoRA：注入（含 FP16 dtype 一致）、注入瞬间零差异、导出、冷启动回读生效
3. 视频超分：预览模式、完整链路、帧数、分辨率预设、音轨保留、时长一致
4. 硬件编码：h264_nvenc / hevc_nvenc 能力探测
5. LoRA 微调：GPU 上跑若干 step，导出后能被推理引擎加载（``--skip-train`` 可跳过）
6. WebUI：Blocks 构建、事件绑定参数个数与控制台处理器直调、退化预览

用法：
    python tests/selftest_e2e.py                 # 全部跑（含一次小规模训练）
    python tests/selftest_e2e.py --skip-train    # 跳过训练，最快
    python tests/selftest_e2e.py --keep          # 保留产物便于排查

退出码 0 表示全部通过。
"""

from __future__ import annotations

import argparse
import inspect
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []
SKIPPED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    if condition:
        PASSED.append(name)
        print(f"  ✅ {name}" + (f"（{detail}）" if detail else ""))
    else:
        FAILED.append(f"{name} {detail}".strip())
        print(f"  ❌ {name} {detail}")
    return bool(condition)


def skip(name: str, reason: str) -> None:
    SKIPPED.append(name)
    print(f"  ⏭  {name}：{reason}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


class PlainProgress:
    """替代 ``gr.Progress``：处理器内部只要求 ``(value, desc=...)`` 可调用。"""

    def __init__(self) -> None:
        self.calls: list[tuple[float, str | None]] = []

    def __call__(self, value: float = 0.0, desc: str | None = None, **_kwargs) -> None:
        self.calls.append((value, desc))


def _make_photo(width: int = 320, height: int = 240):
    """造一张有梯度与噪点的图，便于看出超分是否真的生效。"""
    import numpy as np

    rng = np.random.default_rng(7)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = np.linspace(0, 255, width).astype(np.uint8)[None, :]
    image[:, :, 1] = np.linspace(255, 0, height).astype(np.uint8)[:, None]
    image[:, :, 2] = (rng.random((height, width)) * 255).astype(np.uint8)
    return image


def _make_hr_dataset(directory: Path, count: int = 6, size: int = 192) -> None:
    """合成 HR 训练数据（有结构 + 有细节）。"""
    import cv2
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    for index in range(count):
        image = np.zeros((size, size, 3), dtype=np.uint8)
        image[:, :, 0] = np.linspace(20, 235, size).astype(np.uint8)[None, :]
        image[:, :, 1] = np.linspace(120, 40, size).astype(np.uint8)[:, None]
        image[:, :, 2] = (rng.random((size, size)) * 255).astype(np.uint8)
        for _ in range(24):
            x, y = rng.integers(0, size - 12, 2)
            cv2.circle(image, (int(x), int(y)), int(rng.integers(3, 9)), (255, 255, 255), -1)
        cv2.imwrite(str(directory / f"hr_{index:02d}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _make_clip(path: Path, seconds: float = 1.0) -> bool:
    """生成一段带音轨的测试视频；失败返回 False。"""
    completed = subprocess.run(
        [
            config.FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=25:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


# --------------------------------------------------------------------------- #
# 1. 图片超分
# --------------------------------------------------------------------------- #
def test_image(device: str) -> None:
    section("图片超分（真实权重 + 4K 预设 + FP16）")
    from src import img_sr

    photo = _make_photo()
    started = time.time()
    result, info = img_sr.upscale_image(
        photo,
        preset_name="4K (3840x2160)",
        device=device,          # 故意传 UI 标签
        half=True,
        outscale=4.0,
        auto_download=False,
    )
    elapsed = time.time() - started
    check("输出长边 = 3840", max(result.shape[:2]) == 3840,
          f"{result.shape[1]}x{result.shape[0]}")
    check("设备参数已归一化", info["device"] == config.normalize_device(device),
          f"info.device={info['device']!r}")
    check("耗时在合理范围", 0 < elapsed < 120, f"{elapsed:.2f}s")


# --------------------------------------------------------------------------- #
# 2. LoRA 注入 / 导出 / 回读
# --------------------------------------------------------------------------- #
def test_lora(device: str) -> None:
    section("LoRA（FP16 注入 / 零差异 / 导出 / 冷启动回读）")
    import numpy as np
    import torch

    from src import img_sr, lora_runtime

    lora_path: Path | None = None
    engine = img_sr.get_engine().load(device=device, half=True, auto_download=False)
    try:
        check("主干为 fp16", next(engine.upsampler.model.parameters()).dtype == torch.float16,
              str(next(engine.upsampler.model.parameters()).dtype))

        patch = _make_photo(256, 256)
        before = engine.upscale(patch, outscale=4.0)

        injected = lora_runtime.inject_lora(engine.upsampler.model, rank=4, alpha=4.0)
        check("LoRA 注入成功", len(injected) > 0, f"{len(injected)} 层")

        lora_layer = next(
            m for m in engine.upsampler.model.modules()
            if isinstance(m, lora_runtime.LoRAConv2d)
        )
        check("LoRA 权重跟随主干精度（fp16 下不报 dtype 冲突）",
              lora_layer.lora_A.weight.dtype == torch.float16,
              str(lora_layer.lora_A.weight.dtype))

        after_inject = engine.upscale(patch, outscale=4.0)
        check("B 初始为 0：注入瞬间输出与原始模型逐位一致",
              bool(np.array_equal(before, after_inject)))

        lora_params, total_params = lora_runtime.count_lora_parameters(engine.upsampler.model)
        check("LoRA 参数量远小于总量", 0 < lora_params < total_params * 0.5,
              f"{lora_params:,} / {total_params:,}")

        # 给 B 一点随机值（模拟训练后的权重），再导出并冷启动回读
        for module in engine.upsampler.model.modules():
            if isinstance(module, lora_runtime.LoRAConv2d):
                with torch.no_grad():
                    module.lora_B.weight.normal_(0, 1e-3)

        lora_path = lora_runtime.save_lora_weights(
            engine.upsampler.model,
            config.LORA_DIR / f"selftest_e2e_{int(time.time())}.pth",
            rank=4, alpha=4.0, base_model=config.DEFAULT_MODEL_NAME,
            meta={"source": "selftest_e2e"},
        )
        check("LoRA 已导出", lora_path.is_file(),
              f"{lora_path.stat().st_size / 1024:.0f} KB")

        payload = torch.load(str(lora_path), map_location="cpu", weights_only=False)
        check("导出只含 LoRA 张量",
              all(".lora_A." in k or ".lora_B." in k for k in payload["lora_state_dict"]),
              f"{len(payload['lora_state_dict'])} 个张量")
        check("导出带格式标记", payload.get("format") == "photovideoai-sr-lora")
    finally:
        engine.unload()

    if lora_path is None:
        return

    # 冷启动一个新引擎，按文件名加载（等价于 UI 下拉选择）
    engine2 = img_sr.RealESRGANEngine().load(
        device=device, half=True, lora_selection=lora_path.name, auto_download=False
    )
    try:
        check("引擎可加载该 LoRA", engine2.lora_info is not None,
              f"匹配 {engine2.lora_info['loaded']} 个张量" if engine2.lora_info else "失败")
        with_lora = engine2.upscale(_make_photo(128, 128), outscale=4.0)
        check("带 LoRA 推理可用", with_lora.shape[0] == 512 and with_lora.shape[1] == 512,
              f"{with_lora.shape[1]}x{with_lora.shape[0]}")
    finally:
        engine2.unload()
        lora_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# 3. 视频超分
# --------------------------------------------------------------------------- #
def test_video(device: str, work: Path) -> None:
    section("视频超分（预览 / 完整 / 音轨 / 时长）")
    from src import ffmpeg_utils, video_sr_realesrgan

    try:
        ffmpeg_utils.ensure_ffmpeg()
    except ffmpeg_utils.FFmpegError as exc:
        skip("视频链路", str(exc)[:60])
        return

    source = work / "clip.mp4"
    if not check("生成测试视频", _make_clip(source, seconds=1.0)):
        return
    src_info = ffmpeg_utils.get_video_info(source)

    # 预览模式
    target, info = video_sr_realesrgan.upscale_video(
        source, device=device, preset_name="原始比例 (不限制长边)",
        outscale=4.0, half=True, preview_only=True, preview_frames=6,
        auto_download=False,
    )
    check("预览模式只渲染 6 帧", info["frames"] == 6, f"frames={info['frames']}")
    check("预览输出为 4 倍分辨率", info["output_size"] == "1280x960", info["output_size"])
    out_info = ffmpeg_utils.get_video_info(target)
    check("预览输出可解码", out_info["nb_frames"] == 6, f"nb_frames={out_info['nb_frames']}")

    # 完整链路 + 分辨率预设
    target2, info2 = video_sr_realesrgan.upscale_video(
        source, device=device, preset_name="720p (1280x720)",
        outscale=4.0, half=True, video_format="MP4", auto_download=False,
    )
    out_info2 = ffmpeg_utils.get_video_info(target2)
    check("完整链路处理全部帧", info2["frames"] == src_info["nb_frames"],
          f"{info2['frames']}/{src_info['nb_frames']}")
    check("长边缩放到 1280", max(out_info2["width"], out_info2["height"]) == 1280,
          f"{out_info2['width']}x{out_info2['height']}")
    check("音轨保留", bool(out_info2["has_audio"]) == bool(src_info["has_audio"]),
          f"audio={out_info2['audio_codec']}")
    check("时长与源一致",
          out_info2["duration"] is not None
          and abs(out_info2["duration"] - src_info["duration"]) < 0.3,
          f"src={src_info['duration']}s out={out_info2['duration']}s")


# --------------------------------------------------------------------------- #
# 4. 硬件编码
# --------------------------------------------------------------------------- #
def test_encoders() -> None:
    section("硬件编码器探测")
    from src import ffmpeg_utils

    if not config.ffmpeg_available():
        skip("编码器探测", "未安装 FFmpeg")
        return
    for codec in ("h264_nvenc", "hevc_nvenc"):
        available = ffmpeg_utils.has_nvenc(codec)
        print(f"  ℹ️  {codec}: {'可用' if available else '不可用（将回退软件编码）'}")
    # 这里不断言「必须可用」：没有 NVIDIA 显卡的机器同样可以跑其他用例。
    check("编码器探测不抛异常", isinstance(ffmpeg_utils.has_nvenc("h264_nvenc"), bool))


# --------------------------------------------------------------------------- #
# 4.5 FFmpeg 自动下载（只校验配置与短路逻辑，不联网）
# --------------------------------------------------------------------------- #
def test_ffmpeg_installer() -> None:
    section("FFmpeg 自动下载（配置与幂等，不联网）")
    from src import ffmpeg_installer as installer

    check("平台可识别", installer.platform_key() in {"windows", "linux", "unsupported"},
          installer.platform_key())
    urls = installer.candidate_urls()
    check("候选下载地址与平台一致",
          bool(urls) == installer.is_supported(), f"{len(urls)} 个")
    check("状态文案可生成", bool(installer.status_text()))
    check("安装目录解析正确",
          config.FFMPEG_BUNDLE_DIR.name == "ffmpeg", str(config.FFMPEG_BUNDLE_DIR))

    if not config.ffmpeg_available():
        skip("下载幂等性", "当前没有可用的 FFmpeg（真实下载需要联网，已跳过）")
        return
    # 已可用时 install() 必须直接短路，不能触发下载
    first = installer.install()
    check("已可用时 install() 幂等短路", first.is_file(), str(first))


# --------------------------------------------------------------------------- #
# 4.6 任务取消（协作式）
# --------------------------------------------------------------------------- #
def test_cancel(device: str, work: Path) -> None:
    section("任务取消（令牌 / 视频中途 / 批量中途）")
    import threading

    from src import task_control, video_sr_realesrgan

    # --- 令牌本身 ---
    with task_control.task_scope("自测令牌") as token:
        check("作用域注册后可见", task_control.active_count() == 1,
              f"{task_control.active_count()} 个")
        token.raise_if_cancelled()
        check("未取消时检查不抛异常", True)
        sent = task_control.cancel_all()
        check("cancel_all 返回受影响任务数", sent == 1, f"{sent} 个")
        raised = False
        try:
            token.raise_if_cancelled()
        except task_control.TaskCancelled:
            raised = True
        check("取消后立即抛 TaskCancelled", raised)
    check("作用域退出后自动注销", task_control.active_count() == 0)

    # --- 视频：中途取消 ---
    clip = work / "cancel_clip.mp4"
    if check("生成用于取消测试的视频", _make_clip(clip, seconds=2.0)):
        timer = threading.Timer(1.5, task_control.cancel_all)
        cancelled = False
        try:
            with task_control.task_scope("视频超分（取消测试）") as token:
                timer.start()
                video_sr_realesrgan.upscale_video(
                    clip,
                    device=device,
                    outscale=4.0,
                    half=True,
                    video_format="MP4",
                    auto_download=False,
                    should_cancel=token.raise_if_cancelled,
                )
        except task_control.TaskCancelled:
            cancelled = True
        finally:
            timer.cancel()
        check("视频任务可被中途取消", cancelled)
        leftovers = [p.name for p in config.TEMP_DIR.glob("video_sr_*")]
        check("取消后临时目录已清理", not leftovers, str(leftovers[:3]))

    # --- 批量：中途取消（经 app 处理器，顺带验证面向前端的提示文案） ---
    import app as app_module

    batch_dir = work / "cancel_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    import cv2

    for index in range(6):
        cv2.imwrite(
            str(batch_dir / f"img_{index}.png"),
            cv2.cvtColor(_make_photo(800, 600), cv2.COLOR_RGB2BGR),
        )

    timer = threading.Timer(0.8, task_control.cancel_all)
    message = ""
    cancelled = False
    try:
        timer.start()
        app_module.handle_image_batch(
            None, str(batch_dir), "原始比例 (不限制长边)", device,
            config.DEFAULT_MODEL_NAME, app_module.NO_LORA, True, 4.0, 0, False, "PNG",
            progress=PlainProgress(),
        )
    except Exception as exc:  # noqa: BLE001 - safe_handler 会把它转成 gr.Error
        message = str(exc)
        cancelled = "取消" in message
    finally:
        timer.cancel()
    check("批量任务可被中途取消", cancelled, message[:80])
    check("取消提示对用户可读", "已取消" in message, message[:80])


# --------------------------------------------------------------------------- #
# 5. GPU 微调冒烟
# --------------------------------------------------------------------------- #
def test_training(device: str, work: Path) -> None:
    section("LoRA 微调（GPU 冒烟：训练 → 导出 → 推理加载）")
    from src import img_sr
    from src.train import trainer as train_mod

    if not train_mod.HAS_TORCH:
        skip("微调", "未安装 PyTorch")
        return
    if not config.cuda_available():
        skip("微调", "未检测到 CUDA")
        return

    data_dir = work / "hr"
    _make_hr_dataset(data_dir)
    cfg = train_mod.TrainConfig(
        dataset_paths=[str(data_dir)],
        lora_name=f"selftest_e2e_{int(time.time())}",
        epochs=1,
        batch_size=2,
        patch_size=96,
        learning_rate=5e-4,
        rank=4,
        alpha=4.0,
        val_ratio=0.34,
        val_interval=1,
        val_images=1,
        save_interval=1,
        device=device,      # 故意传 UI 标签
        num_workers=0,
        seed=0,
        log_every=1,
    )
    cfg.validate()
    check("训练配置归一化 device", cfg.device == config.normalize_device(device),
          f"cfg.device={cfg.device!r}")

    trainer = train_mod.Trainer(cfg)
    try:
        result = trainer.run()
        check("训练产出 LoRA 文件", Path(result["output_path"]).is_file(),
              f"{result['final_size_mb']} MB")
        check("训练确实在 GPU 上", trainer.model is not None
              and next(trainer.model.parameters()).device.type == "cuda")
        check("验证 PSNR 有记录", result["best_psnr"] is not None, f"{result['best_psnr']} dB")

        engine = img_sr.RealESRGANEngine().load(
            device=device, half=True,
            lora_selection=Path(result["output_path"]).name, auto_download=False,
        )
        try:
            check("推理引擎可加载训练产物", engine.lora_info is not None,
                  f"匹配 {engine.lora_info['loaded']} 个张量" if engine.lora_info else "失败")
        finally:
            engine.unload()
    finally:
        # 清理本次训练产物，避免污染 lora_weights/
        for path in config.LORA_DIR.glob(f"{cfg.lora_name}*"):
            path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# 6. WebUI
# --------------------------------------------------------------------------- #
def test_ui(device: str, work: Path) -> None:
    section("WebUI（Blocks 构建 / 事件接线 / 处理器直调）")
    try:
        import app
    except Exception as exc:  # noqa: BLE001
        skip("WebUI", f"导入 app.py 失败：{exc}")
        return
    if not app.HAS_GRADIO:
        skip("WebUI", "未安装 Gradio")
        return

    interface = app.build_interface()
    check("build_interface 成功", interface is not None)

    # 反查事件绑定：处理器签名（去掉 progress）应与绑定的输入个数一致
    mismatches: list[str] = []
    checked = 0
    for block_fn in interface.fns.values():
        fn = getattr(block_fn, "fn", None)
        if not callable(fn):
            continue
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        params = list(signature.parameters.values())
        if not params or any(
            p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params
        ):
            continue
        bound = getattr(block_fn, "inputs", None) or []
        if not bound:
            continue
        checked += 1
        expected = len([p for p in params if p.name != "progress"])
        if expected != len(bound):
            mismatches.append(
                f"{getattr(fn, '__name__', fn)} 签名 {expected} vs 绑定 {len(bound)}"
            )
    check("事件绑定与处理器签名一致", not mismatches,
          f"检查 {checked} 个绑定" + ("；" + "; ".join(mismatches) if mismatches else ""))

    # 直接调用照片 Tab 的处理器（等价于点按钮）
    progress = PlainProgress()
    image, info_md = app.handle_image(
        _make_photo(), "1080p (1920x1080)", device, config.DEFAULT_MODEL_NAME,
        app.NO_LORA, True, 4.0, 0, False, False, "PNG",
        False, 5.0, False, False,
        progress=progress,
    )
    check("照片 Tab 处理器可用", getattr(image, "size", 0) > 0
          and max(image.shape[:2]) == 1920, f"{image.shape[1]}x{image.shape[0]}")
    check("任务信息面板已生成", "输出分辨率" in info_md)
    check("进度回调被调用", bool(progress.calls), f"{len(progress.calls)} 次")

    # 退化预览（微调 Tab）
    data_dir = work / "hr"
    if data_dir.is_dir() and app.HAS_TRAIN:
        paths, _note = app.handle_preview_degradation(None, str(data_dir), 128, 4, 2, 0)
        check("退化预览生成 2 组", len(paths) == 2, f"{len(paths)} 组")

    # 「下载 FFmpeg」按钮：已可用时必须短路（不联网），且返回个数要与界面接线一致
    result = app.handle_install_ffmpeg(progress=PlainProgress())
    check("下载 FFmpeg 处理器返回 4 个输出（与接线一致）",
          isinstance(result, tuple) and len(result) == 4,
          f"{len(result)} 个输出" if isinstance(result, tuple) else type(result).__name__)
    if isinstance(result, tuple) and result:
        check("FFmpeg 已可用时不重复下载", "无需重复下载" in str(result[0]))


# --------------------------------------------------------------------------- #
# 7. 释放显存
# --------------------------------------------------------------------------- #
def test_release_vram(device: str, work: Path) -> None:
    section("释放显存（卸载模型 + 回收 CUDA 缓存）")
    import app

    from src import img_sr, task_control
    from src.common import get_free_vram_gb

    # 先制造真实占用：加载模型 + 一次 4K 推理会留下大量缓存池
    img_sr.upscale_image(
        _make_photo(640, 480),
        preset_name="4K (3840x2160)",
        device=device,
        half=True,
        outscale=4.0,
        auto_download=False,
    )
    check("推理后引擎处于已加载状态", img_sr.get_engine().ready)
    before = get_free_vram_gb(0)

    message = app.handle_release_vram()
    after = get_free_vram_gb(0)

    check("释放后引擎已卸载", not img_sr.get_engine().ready)
    check("提示里报告了模型与显存", "卸载的模型" in message and "空闲显存" in message)
    if before is not None and after is not None:
        check("空闲显存确有增加", after >= before - 0.05,
              f"{before:.2f} → {after:.2f} GB")

    # 有任务在跑时必须拒绝，否则会把正在推理的模型卸掉
    with task_control.task_scope("占位任务"):
        warning = app.handle_release_vram()
    check("有任务运行时拒绝释放", "正在运行" in warning, warning[:60])

    # 释放后应能立即继续工作（按需重新加载）
    result, _info = img_sr.upscale_image(
        _make_photo(128, 96), device=device, half=True, outscale=4.0, auto_download=False
    )
    check("释放后仍可继续推理", result.shape[:2] == (384, 512),
          f"{result.shape[1]}x{result.shape[0]}")


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PhotoVideoAI-SR 端到端自测")
    parser.add_argument("--skip-train", action="store_true", help="跳过 GPU 微调冒烟")
    parser.add_argument("--keep", action="store_true", help="保留临时产物")
    args = parser.parse_args(argv)

    work = config.TEMP_DIR / f"selftest_e2e_{time.strftime('%Y%m%d_%H%M%S')}"
    work.mkdir(parents=True, exist_ok=True)

    print("PhotoVideoAI-SR 端到端自测")
    print(config.runtime_summary())
    print(f"临时目录：{work}")

    from src import img_sr

    if not config.cuda_available():
        print("\n⏭  未检测到可用 CUDA，端到端自测需要 NVIDIA GPU，已跳过。")
        print("   只想验证代码与媒体链路，请运行：python tests/selftest.py")
        return 0
    if not img_sr.HAS_REALESRGAN:
        print(f"\n❌ RealESRGAN 依赖不可用：{img_sr.REALESRGAN_IMPORT_ERROR}")
        return 1
    weights = img_sr.model_path(config.DEFAULT_MODEL_NAME)
    if not weights.is_file():
        print(f"\n⏭  缺少基础权重 {weights}，已跳过。")
        print("   可从 README「模型下载」章节获取，或让程序自动下载。")
        return 0

    device = config.get_gpu_options()[0]
    started = time.time()
    try:
        test_image(device)
        test_lora(device)
        test_video(device, work)
        test_encoders()
        test_ffmpeg_installer()
        test_cancel(device, work)
        if args.skip_train:
            skip("LoRA 微调", "--skip-train")
        else:
            test_training(device, work)
        test_ui(device, work)
        test_release_vram(device, work)
    finally:
        elapsed = time.time() - started
        print("\n" + "=" * 60)
        print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项，跳过 {len(SKIPPED)} 项"
              f"（总耗时 {elapsed:.1f}s）")
        for item in FAILED:
            print(f"  - 失败：{item}")
        for item in SKIPPED:
            print(f"  - 跳过：{item}")
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"临时产物保留在：{work}")
        print(f"推理结果按程序默认行为写入 {config.OUTPUT_DIR}（已在 .gitignore 中忽略）")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
