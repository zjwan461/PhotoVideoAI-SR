#!/usr/bin/env python3
"""PhotoVideoAI-SR — Gradio WebUI 主入口。

定位：图片 / 视频 AI 超分（NVIDIA CUDA only），FFmpeg 负责媒体编解码。
本文件只做 UI 与调度，真正的推理逻辑都在 ``src/`` 下的独立模块里。

启动：
    python app.py                    # 默认 127.0.0.1:7860
    python app.py --host 0.0.0.0 --port 8000 --share
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
import threading
import time
import traceback
from pathlib import Path

# 保证以任意工作目录运行时都能 import 到项目模块
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import (  # noqa: E402
    enhance_extras,  # noqa: E402
    ffmpeg_installer,
    ffmpeg_utils,
    img_sr,
)
from src import video_sr_maxine as maxine_mod  # noqa: E402
from src import video_sr_realesrgan as rr_mod  # noqa: E402
from src.common import (  # noqa: E402
    check_vram_for_task,
    cleanup_stale_temp,
    friendly_cuda_error,
    get_device_summary,
    get_logger,
    list_image_files,
    log_exception,
    log_runtime_environment,
    log_task,
    lora_choices,
    scan_lora_weights,
    unique_output_path,
)

# 微调模块（可选高级功能）：缺 PyTorch / 依赖时置为 None，UI 自动降级
try:
    from src.train import lora_utils as train_lora_utils
    from src.train import trainer as train_mod

    HAS_TRAIN = train_mod.HAS_TORCH
    TRAIN_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    train_mod = None  # type: ignore[assignment]
    train_lora_utils = None  # type: ignore[assignment]
    HAS_TRAIN = False
    TRAIN_IMPORT_ERROR = str(exc)

logger = get_logger("app")

try:
    import gradio as gr

    HAS_GRADIO = True
except Exception as exc:  # pragma: no cover
    gr = None  # type: ignore[assignment]
    HAS_GRADIO = False
    GRADIO_IMPORT_ERROR = str(exc)

NO_LORA = "不使用 LoRA"

CSS = """
.sr-title { text-align: center; }
.sr-sub { text-align: center; color: #666; margin-bottom: 8px; }
.sr-info textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.sr-warn { color: #b45309; }
"""


# --------------------------------------------------------------------------- #
# 错误处理封装
# --------------------------------------------------------------------------- #
def safe_handler(task_name: str):
    """把处理器异常转成友好提示，避免 Web 服务崩溃。"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except gr.Error:
                raise
            except (
                img_sr.ImageSRError,
                rr_mod.VideoSRError,
                maxine_mod.MaxineError,
                ffmpeg_utils.FFmpegError,
                ffmpeg_installer.FFmpegInstallError,
            ) as exc:
                # 这些异常的 message 已经是面向用户的文案，原样展示
                log_exception(task_name, exc)
                raise gr.Error(str(exc)) from None
            except RuntimeError as exc:
                friendly = friendly_cuda_error(exc)
                log_exception(task_name, exc)
                raise gr.Error(friendly or f"{task_name}失败：{exc}") from None
            except Exception as exc:  # noqa: BLE001
                log_exception(task_name, exc)
                logger.debug(traceback.format_exc())
                raise gr.Error(f"{task_name}发生未预期错误：{exc}（详情见 logs/ 目录）") from None

        return wrapper

    return decorator


# --------------------------------------------------------------------------- #
# 通用 UI 片段
# --------------------------------------------------------------------------- #
def info_panel_html(info: dict) -> str:
    """把任务信息字典渲染成 Markdown 信息面板。"""
    if not info:
        return "_暂无任务信息_"
    rows = [
        ("耗时", f"{info.get('elapsed', 0):.2f} 秒"),
        ("输入分辨率", info.get("input_size", "-")),
        ("输出分辨率", info.get("output_size", "-")),
        ("模型", info.get("model", "-")),
        ("LoRA", info.get("lora", "-")),
        ("设备", info.get("device", "-")),
        ("帧数", info.get("frames", "-")),
        ("输出帧率", f"{info.get('fps')} fps" if info.get("fps") else "-"),
        ("时长", f"{info.get('duration')} 秒" if info.get("duration") else "-"),
        ("音频", info.get("audio", "-")),
    ]
    free_vram = info.get("free_vram_gb")
    if free_vram is not None:
        rows.append(("任务后空闲显存", f"{free_vram} GB"))
    body = "\n".join(f"| {key} | {value} |" for key, value in rows)
    return "| 项目 | 值 |\n| --- | --- |\n" + body


def environment_markdown() -> str:
    """环境状态面板内容。"""
    return "```\n" + config.runtime_summary() + "\n```"


def encoder_markdown() -> str:
    """编码器可用性面板内容。"""
    return "```\n" + ffmpeg_utils.probe_encoder_support() + "\n```"


def lora_markdown() -> str:
    """LoRA 列表面板内容。"""
    items = scan_lora_weights()
    if not items:
        return (
            f"`{config.LORA_DIR}` 下暂无 LoRA 权重。\n\n"
            "把训练好的 `*.pth` 放进该目录，刷新后即可在图片/视频 Tab 的下拉框中选择。"
        )
    lines = [f"共 {len(items)} 个 LoRA："]
    lines += [f"- `{name}`" for name in items]
    return "\n".join(lines)


def refresh_lora_choices():
    """刷新各 Tab 的 LoRA 下拉框选项。"""
    return gr.update(choices=lora_choices(), value=NO_LORA)


@safe_handler("下载 FFmpeg")
def handle_install_ffmpeg(progress=gr.Progress()):
    """下载 FFmpeg 到项目内并立即接管（无需重启、无需配置 PATH）。

    给没有预装 FFmpeg 的用户准备的兜底：成功后 ``config.FFMPEG_BIN`` 会指向
    项目内 ``tools/ffmpeg/`` 里的可执行文件，视频链路马上可用。
    """
    from src import ffmpeg_installer

    if config.ffmpeg_available():
        return (
            ffmpeg_installer.status_text() + "\n\n无需重复下载。",
            ffmpeg_installer.status_text(),
            environment_markdown(),
            encoder_markdown(),
        )

    path = ffmpeg_installer.install(
        progress=lambda value, message: progress(value, desc=message)
    )
    note = (
        f"✅ FFmpeg 已安装到项目内：`{path}`\n\n"
        "后续任务会直接使用它，**无需重启、也无需修改系统 PATH**。\n\n"
        f"如想改用系统自带的 FFmpeg，删掉 `{config.FFMPEG_BUNDLE_DIR}` 即可恢复为从 PATH 查找。"
    )
    return (
        note,
        ffmpeg_installer.status_text(),
        environment_markdown(),
        encoder_markdown(),
    )


# --------------------------------------------------------------------------- #
# Tab1：图片超分
# --------------------------------------------------------------------------- #
@safe_handler("图片超分")
def handle_image(
    image,
    preset_name,
    device,
    model_name,
    lora_selection,
    half_precision,
    outscale,
    tile_size,
    auto_download,
    face_restore,
    image_format,
    denoise_enabled=False,
    denoise_strength=5.0,
    scratch_enabled=False,
    colorize_enabled=False,
    progress=gr.Progress(),
):
    """单张图片超分处理器（含可选的老照片附加增强）。"""
    if image is None:
        raise gr.Error("请先上传一张图片。")

    target_long_edge = img_sr.resolve_target_long_edge(preset_name)
    ok, message = check_vram_for_task(
        img_sr.estimate_required_vram_gb(target_long_edge, outscale), device, "图片超分"
    )
    if not ok:
        logger.warning(message)
    else:
        logger.info(message)

    extras_applied: list[str] = []
    prepared = image
    # 去噪 / 去划痕属于「输入侧修复」，放在超分之前做，避免放大噪声
    try:
        if denoise_enabled or scratch_enabled:
            prepared = enhance_extras.apply_extras(
                prepared,
                denoise=bool(denoise_enabled),
                denoise_strength=float(denoise_strength),
                scratch=bool(scratch_enabled),
            )
            if denoise_enabled:
                extras_applied.append(f"去噪({denoise_strength:g})")
            if scratch_enabled:
                extras_applied.append("去划痕")
    except enhance_extras.ExtrasError as exc:
        raise gr.Error(str(exc)) from None

    result, info = img_sr.upscale_image(
        prepared,
        preset_name=preset_name,
        device=device,
        model_name=model_name,
        lora_selection=lora_selection,
        half=bool(half_precision),
        outscale=float(outscale),
        tile=int(tile_size) or None,
        auto_download=bool(auto_download),
        face_restore=bool(face_restore),
        progress=lambda value, text: progress(value, desc=text),
    )

    # 上色属于「输出侧」，放在超分之后
    if colorize_enabled:
        try:
            result = enhance_extras.colorize_image(result)
            extras_applied.append(f"上色({enhance_extras.colorizer_name()})")
        except enhance_extras.ExtrasError as exc:
            raise gr.Error(str(exc)) from None

    # 同时落盘一份，方便用户直接取用
    suffix = ".jpg" if str(image_format).upper() == "JPG" else ".png"
    target = unique_output_path("image_sr", suffix)
    try:
        img_sr.write_image(target, result)
        info["output_path"] = str(target)
        saved_note = f"\n\n已保存到：`{target}`"
    except img_sr.ImageSRError as exc:
        logger.warning("结果保存失败：%s", exc)
        saved_note = f"\n\n⚠️ 结果保存失败：{exc}"

    if extras_applied:
        saved_note += "\n\n附加处理：" + "、".join(extras_applied)

    log_task("图片超分", {"model": model_name, "preset": preset_name, "lora": lora_selection},
             info["elapsed"], get_device_summary(device))
    return result, info_panel_html(info) + saved_note


@safe_handler("批量图片超分")
def handle_image_batch(
    files,
    folder_path,
    preset_name,
    device,
    model_name,
    lora_selection,
    half_precision,
    outscale,
    tile_size,
    auto_download,
    image_format,
    progress=gr.Progress(),
):
    """批量图片超分：优先使用上传的多个文件，其次使用服务器上的文件夹路径。"""
    inputs: list[Path] = []
    if files:
        inputs = [Path(f if isinstance(f, str) else getattr(f, "name", str(f))) for f in files]
    elif folder_path and str(folder_path).strip():
        inputs = list_image_files(str(folder_path).strip())
        if not inputs:
            raise gr.Error(f"该目录下没有找到图片：{folder_path}")
    if not inputs:
        raise gr.Error("请上传多张图片，或填写服务器上包含图片的文件夹路径。")

    output_dir = config.OUTPUT_DIR / "batch"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    failures: list[str] = []
    started = time.time()

    for index, path in enumerate(inputs, start=1):
        progress((index - 1) / len(inputs), desc=f"[{index}/{len(inputs)}] {Path(path).name}")
        try:
            target, _info = _upscale_single_to_dir(
                Path(path), output_dir, preset_name, device, model_name,
                lora_selection, half_precision, outscale, tile_size, auto_download, image_format,
            )
            outputs.append(str(target))
        except Exception as exc:  # noqa: BLE001
            log_exception(f"批量图片超分 {Path(path).name}", exc)
            failures.append(f"{Path(path).name}: {exc}")

    progress(1.0, desc="批量完成")
    elapsed = time.time() - started
    summary = (
        f"### 批量处理完成\n"
        f"- 总数：{len(inputs)}\n- 成功：{len(outputs)}\n- 失败：{len(failures)}\n"
        f"- 耗时：{elapsed:.1f} 秒\n- 输出目录：`{output_dir}`"
    )
    if failures:
        summary += "\n\n**失败明细：**\n" + "\n".join(f"- {item}" for item in failures[:20])
    log_task("批量图片超分", {"count": len(inputs)}, elapsed, get_device_summary(device))
    return outputs, summary


def _upscale_single_to_dir(
    input_path: Path, output_dir: Path, preset_name, device, model_name,
    lora_selection, half_precision, outscale, tile_size, auto_download, image_format,
) -> tuple[Path, dict]:
    """单图超分并把结果写入指定目录。"""
    image = img_sr.read_image(input_path)
    result, info = img_sr.upscale_image(
        image,
        preset_name=preset_name,
        device=device,
        model_name=model_name,
        lora_selection=lora_selection,
        half=bool(half_precision),
        outscale=float(outscale),
        tile=int(tile_size) or None,
        auto_download=bool(auto_download),
    )
    suffix = ".jpg" if str(image_format).upper() == "JPG" else ".png"
    target = output_dir / f"{input_path.stem}_sr{suffix}"
    counter = 1
    while target.exists():
        target = output_dir / f"{input_path.stem}_sr_{counter}{suffix}"
        counter += 1
    img_sr.write_image(target, result)
    info["output_path"] = str(target)
    return target, info


# --------------------------------------------------------------------------- #
# Tab2：视频超分
# --------------------------------------------------------------------------- #
@safe_handler("视频超分")
def handle_video(
    video_path,
    preset_name,
    device,
    video_model,
    lora_selection,
    half_precision,
    outscale,
    tile_size,
    auto_download,
    video_format,
    preview_only,
    preview_frames,
    fps_override,
    quality_mode,
    bitrate_mbps,
    crf,
    progress=gr.Progress(),
):
    """视频超分处理器：按所选模型分派到 RealESRGAN 或 Maxine 链路。"""
    if not video_path:
        raise gr.Error("请先上传一段视频。")
    if not ffmpeg_utils.check_ffmpeg().startswith("✅"):
        raise gr.Error(ffmpeg_utils.check_ffmpeg())

    use_maxine = str(video_model).startswith("Maxine")
    if use_maxine and not maxine_mod.HAS_MAXINE:
        raise gr.Error("当前环境不可用 Maxine VSR（未安装 nvvfx）。请改用 RealESRGAN 链路。")

    fps_value = float(fps_override) if fps_override else None
    bitrate = float(bitrate_mbps) if quality_mode == "固定码率" and bitrate_mbps else None
    crf_value = int(crf) if quality_mode == "恒定质量" and crf else None

    if not use_maxine:
        ok, message = check_vram_for_task(
            rr_mod.estimate_required_vram_gb(img_sr.resolve_target_long_edge(preset_name), outscale),
            device,
            "视频逐帧超分",
        )
        logger.info(message if ok else message)

    common_kwargs = dict(
        preset_name=preset_name,
        device=device,
        video_format=video_format,
        preview_only=bool(preview_only),
        preview_frames=int(preview_frames or config.PREVIEW_FRAME_LIMIT),
        fps_override=fps_value,
        bitrate_mbps=bitrate,
        crf=crf_value,
        progress=lambda value, text: progress(value, desc=text),
    )

    if use_maxine:
        target, info = maxine_mod.upscale_video_maxine(video_path, **common_kwargs)
        task_label = "Maxine 视频超分"
    else:
        target, info = rr_mod.upscale_video(
            video_path,
            model_name=model_name_from_selection(video_model),
            lora_selection=lora_selection,
            half=bool(half_precision),
            outscale=float(outscale),
            tile=int(tile_size) or None,
            auto_download=bool(auto_download),
            **common_kwargs,
        )
        task_label = "视频逐帧超分"

    log_task(task_label, {"model": video_model, "preview": preview_only, "format": video_format},
             info["elapsed"], get_device_summary(device))
    return str(target), info_panel_html(info)


def model_name_from_selection(selection: str | None) -> str:
    """视频模型下拉框 -> 基础模型文件名。"""
    models = img_sr.available_base_models()
    if not selection:
        return models[0]
    for name in models:
        if name in str(selection):
            return name
    return models[0]


def preview_video_info(video_path):
    """上传视频后立即展示源信息。"""
    if not video_path:
        return "_上传视频后显示媒体信息_"
    try:
        info = ffmpeg_utils.get_video_info(video_path)
        return "**源视频信息：** " + ffmpeg_utils.format_video_info(info)
    except ffmpeg_utils.FFmpegError as exc:
        return f"⚠️ 无法读取视频信息：{exc}"


# --------------------------------------------------------------------------- #
# Tab3：LoRA 微调（可选高级功能）
# --------------------------------------------------------------------------- #
#: 当前正在运行/最近一次的训练任务（同一时间只允许一个）
_ACTIVE_RUNNER = None
_TRAIN_LOCK = threading.Lock()


def training_status_markdown() -> str:
    """微调 Tab 顶部的环境与依赖状态。"""
    if not HAS_TRAIN:
        return (
            "### ⚠️ 微调功能不可用\n\n"
            "未检测到可用的 PyTorch，训练模块已自动隐藏/降级。\n\n"
            f"```\n{TRAIN_IMPORT_ERROR or 'torch 未安装'}\n```\n"
            "安装 CUDA 版 PyTorch 后重启即可启用（主推理链路不受影响）。"
        )
    colorize_state = (
        f"✅ {enhance_extras.colorizer_name()}"
        if enhance_extras.colorizer_available()
        else "❌ 未注册（图片 Tab 的上色开关已隐藏）"
    )
    extra = [
        f"感知损失 (lpips)：{'✅ 可用' if train_mod.HAS_LPIPS else '❌ 未安装，将只用 L1'}",
        f"上色后端：{colorize_state}",
        f"续训检查点：{max(0, len(train_mod.default_resume_candidates()) - 1)} 个可用",
        f"基础模型：{', '.join(img_sr.available_base_models())}",
    ]
    return "### 🧪 微调模块状态\n\n" + "\n".join(f"- {line}" for line in extra if line)


def lora_files_markdown() -> str:
    """已训练 LoRA 列表。"""
    if not HAS_TRAIN:
        return "_微调模块不可用_"
    entries = train_lora_utils.list_loras()
    if not entries:
        return f"`{config.LORA_DIR}` 下暂无 LoRA 权重。"
    lines = [f"共 {len(entries)} 个："]
    lines += [f"- `{item['name']}`（{item['size_mb']} MB）" for item in entries]
    return "\n".join(lines)


def _collect_training_inputs(files, folder_path):
    """把手写路径与上传文件合并成数据集输入列表。"""
    inputs: list[str] = []
    if folder_path and str(folder_path).strip():
        for chunk in str(folder_path).replace(";", "\n").replace(",", "\n").splitlines():
            chunk = chunk.strip()
            if chunk:
                inputs.append(chunk)
    if files:
        for item in files:
            path = item if isinstance(item, str) else getattr(item, "name", str(item))
            if path:
                inputs.append(str(path))
    return inputs


@safe_handler("退化效果预览")
def handle_preview_degradation(files, folder_path, patch_size, rank_scale, count, seed):
    """开训前预览「HR -> LR 退化」效果，确认退化强度是否合适。"""
    if not HAS_TRAIN:
        raise gr.Error("微调模块不可用：请先安装 CUDA 版 PyTorch。")
    inputs = _collect_training_inputs(files, folder_path)
    if not inputs:
        raise gr.Error("请先指定高清图片文件夹，或上传若干张 HR 图片。")
    paths = train_mod.preview_pairs(
        inputs,
        patch_size=int(patch_size),
        scale=int(rank_scale),
        count=int(count),
        seed=int(seed),
    )
    return paths, f"已生成 {len(paths)} 组退化对比图（左：原图，右：退化后放大）。"


@safe_handler("启动微调")
def handle_start_training(
    files,
    folder_path,
    lora_name,
    base_model,
    device,
    epochs,
    batch_size,
    patch_size,
    learning_rate,
    lora_rank,
    lora_alpha,
    val_ratio,
    val_interval,
    save_interval,
    loss_type,
    perceptual_weight,
    grad_accum,
    use_amp,
    resume_from,
    seed,
):
    """组装配置并启动后台训练。"""
    global _ACTIVE_RUNNER
    if not HAS_TRAIN:
        raise gr.Error("微调模块不可用：请先安装 CUDA 版 PyTorch。")
    if not config.cuda_available() and os.environ.get("PVSR_ALLOW_CPU_TRAIN", "0") not in ("1", "true", "True"):
        raise gr.Error(
            "微调需要使用 CUDA 设备（当前未检测到）。\n"
            "正式训练请在有显卡的机器上进行；仅做冒烟测试时可用环境变量 "
            "PVSR_ALLOW_CPU_TRAIN=1 强制启动。"
        )

    inputs = _collect_training_inputs(files, folder_path)
    if not inputs:
        raise gr.Error("请先指定高清训练数据（文件夹路径或上传图片）。")

    with _TRAIN_LOCK:
        if _ACTIVE_RUNNER is not None and _ACTIVE_RUNNER.running:
            raise gr.Error("已有训练任务在运行中，请先点击「停止训练」或等待其完成。")

        resolve = "(不续训)"
        cfg = train_mod.TrainConfig(
            dataset_paths=inputs,
            lora_name=str(lora_name or "my_lora"),
            epochs=int(epochs),
            batch_size=int(batch_size),
            patch_size=int(patch_size),
            learning_rate=float(learning_rate),
            rank=int(lora_rank),
            alpha=float(lora_alpha) if lora_alpha else None,
            val_ratio=float(val_ratio),
            val_interval=int(val_interval),
            save_interval=int(save_interval),
            loss=str(loss_type),
            perceptual_weight=float(perceptual_weight),
            grad_accum=int(grad_accum),
            amp=bool(use_amp),
            resume_from=None if str(resume_from).startswith(resolve) or not resume_from else str(resume_from),
            base_model=base_model or config.DEFAULT_MODEL_NAME,
            device=device,
            seed=int(seed),
        )
        try:
            cfg.validate()
        except train_mod.TrainingError as exc:
            raise gr.Error(str(exc)) from None

        need_gb = train_mod.estimate_training_vram_gb(cfg.patch_size, cfg.batch_size, cfg.rank)
        ok, message = check_vram_for_task(need_gb, device, "LoRA 微调")
        logger.info(message if ok else message)
        precheck_note = "" if ok else f"\n\n{message}"

        runner = train_mod.TrainingRunner(cfg)
        _ACTIVE_RUNNER = runner
        runner.start()

    status = (
        f"### 🚀 训练已启动\n\n"
        f"- LoRA 名称：`{cfg.lora_name}`\n- 基础模型：`{cfg.base_model}`\n"
        f"- epoch / batch / patch：{cfg.epochs} / {cfg.batch_size} / {cfg.patch_size}\n"
        f"- rank / alpha：{cfg.rank} / {cfg.alpha or cfg.rank}\n- 学习率：{cfg.learning_rate}\n"
        f"- 预估显存：约 {need_gb} GB"
        + precheck_note
    )
    return status, runner.render_log(), runner.render_status()


@safe_handler("停止微调")
def handle_stop_training():
    """请求停止当前训练（会先保存当前进度）。"""
    global _ACTIVE_RUNNER
    if _ACTIVE_RUNNER is None:
        return "当前没有正在运行的训练任务。"
    if not _ACTIVE_RUNNER.running:
        return "训练已经结束。"
    _ACTIVE_RUNNER.stop()
    return "已发送停止请求，训练会在当前 step 结束后保存并退出。"


def stream_training_status(poll_seconds: float = 1.5, max_seconds: float = 8 * 3600):
    """轮询训练状态（Gradio 生成器）：边训练边刷新状态、日志与验证图。"""
    started = time.time()
    if _ACTIVE_RUNNER is None:
        yield ("_尚未启动训练_", "_等待训练启动…_", [], None,
               gr.update(interactive=True), gr.update(interactive=False))
        return
    runner = _ACTIVE_RUNNER
    while True:
        status = runner.status()
        gallery = _latest_validation_images(runner) if status.get("step") else []
        output = status.get("output")
        running = runner.running
        # 运行中禁止重复启动，并允许点击停止
        yield (
            runner.render_status(),
            runner.render_log(),
            gallery,
            output,
            gr.update(interactive=not running),
            gr.update(interactive=running),
        )
        if not running:
            break
        if time.time() - started > max_seconds:
            break
        time.sleep(max(0.5, float(poll_seconds)))


def _latest_validation_images(runner, limit: int = 4) -> list[str]:
    """取当前训练任务的验证图（按 mtime 倒序）。"""
    if runner is None or train_mod is None:
        return []
    try:
        directory = Path(config.LOG_DIR) / "validate" / runner.cfg.run_tag
    except Exception:  # noqa: BLE001
        return []
    if not directory.exists():
        return []
    files = sorted(directory.glob("epoch_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files[:limit]]


@safe_handler("删除 LoRA")
def handle_delete_lora(name):
    """删除 lora_weights/ 中的某个 LoRA。"""
    if not HAS_TRAIN:
        raise gr.Error("微调模块不可用。")
    if not name or str(name).startswith("不使用"):
        raise gr.Error("请选择要删除的 LoRA 文件。")
    if train_lora_utils.delete_lora(name):
        return f"已删除 `{name}`。", lora_files_markdown(), refresh_lora_choices(), refresh_lora_choices()
    raise gr.Error(f"删除失败：{name}（文件不存在或不在 lora_weights/ 内）")


def refresh_train_button_state():
    """按是否已有训练在跑来切换按钮可用性。"""
    running = _ACTIVE_RUNNER is not None and _ACTIVE_RUNNER.running
    return gr.update(interactive=not running), gr.update(interactive=running)


# --------------------------------------------------------------------------- #
# 构建界面
# --------------------------------------------------------------------------- #
def gradio_major_version() -> int:
    """解析 Gradio 主版本号（解析失败按 4 处理）。"""
    try:
        return int(str(gr.__version__).split(".", 1)[0])
    except Exception:  # noqa: BLE001
        return 4


def build_interface() -> gr.Blocks:
    """组装 Gradio 界面（Blocks 组件树），供 :func:`main` 启动。"""
    if not HAS_GRADIO:
        raise SystemExit(
            "未安装 Gradio，无法启动 WebUI。\n"
            f"请执行：pip install gradio\n导入错误：{GRADIO_IMPORT_ERROR}"
        )

    gpu_options = config.get_gpu_options()
    default_device = gpu_options[0]
    model_options = img_sr.available_base_models()
    video_models = config.video_model_options()
    lora_items = lora_choices()

    # Gradio 6 起 css / theme 从 Blocks 构造函数移到 launch()，这里做版本兼容
    style_in_launch = gradio_major_version() >= 6
    blocks_kwargs = {"title": "PhotoVideoAI-SR"}
    if not style_in_launch:
        blocks_kwargs.update({"css": CSS, "theme": gr.themes.Soft()})

    with gr.Blocks(**blocks_kwargs) as interface:
        gr.Markdown("# PhotoVideoAI-SR\n图片 / 视频 AI 超分（NVIDIA CUDA + FFmpeg）", elem_classes="sr-title")
        gr.Markdown(
            "本地运行 · 不上传云端 · MIT 开源。RealESRGAN 负责图片与逐帧视频超分（支持 LoRA 自定义），"
            "Maxine VSR 为可选时序超分组件。",
            elem_classes="sr-sub",
        )

        # ---------------- 全局控件 ---------------- #
        with gr.Accordion("⚙️ 全局设置", open=True):
            with gr.Row():
                device_dd = gr.Dropdown(
                    choices=gpu_options, value=default_device, label="CUDA 设备",
                    info="运行时动态探测，不写死任何硬件型号",
                )
                preset_dd = gr.Dropdown(
                    choices=list(config.RESOLUTION_PRESETS.keys()),
                    value=config.DEFAULT_RESOLUTION_PRESET,
                    label="输出分辨率预设",
                )
                model_dd = gr.Dropdown(
                    choices=model_options, value=model_options[0], label="基础模型",
                    info="权重放在 model_weights/ 目录",
                )
            with gr.Row():
                half_cb = gr.Checkbox(value=config.HALF_PRECISION, label="半精度推理 (FP16)",
                                      info="显存更省、速度更快；出现异常可关闭")
                tile_sl = gr.Slider(0, 1024, value=0, step=32, label="分块大小 tile",
                                    info="0 = 按空闲显存自动选择；显存紧张时手动调小")
                outscale_sl = gr.Slider(1.0, 4.0, value=4.0, step=0.5, label="超分倍率",
                                        info="基础模型固有倍率，随后再缩放到目标分辨率")
            auto_download_cb = gr.Checkbox(value=True, label="自动下载缺失的基础模型",
                                           info="从官方地址下载到 model_weights/，仅首次需要网络")

        with gr.Tabs():
            # ---------------- Tab1 图片 ---------------- #
            with gr.Tab("📷 照片超分"):
                with gr.Row():
                    with gr.Column():
                        image_in = gr.Image(type="numpy", label="输入图片", height=340)
                        with gr.Row():
                            lora_dd_img = gr.Dropdown(
                                choices=lora_items, value=NO_LORA, label="LoRA 权重",
                                info="可选；不选择则使用原生基础模型",
                            )
                            img_fmt_dd = gr.Dropdown(
                                choices=list(config.IMAGE_FORMATS), value=config.DEFAULT_IMAGE_FORMAT,
                                label="输出格式",
                            )
                        face_cb = gr.Checkbox(
                            value=False, label="老照片附加：人脸修复 (GFPGAN)",
                            info="可选依赖，未安装时会自动提示",
                            visible=config.gfpgan_available(),
                        )
                        with gr.Accordion("🧹 附加增强（可选，传统算法，不需要额外权重）", open=False):
                            with gr.Row():
                                denoise_cb = gr.Checkbox(
                                    value=False, label="去噪", info="超分前对输入做非局部均值去噪"
                                )
                                denoise_sl = gr.Slider(
                                    1, 15, value=5, step=1, label="去噪强度",
                                    info="越大越平滑，也越容易糊掉细节",
                                )
                            scratch_cb = gr.Checkbox(
                                value=False, label="轻量去划痕 / 去尘点",
                                info="中值 + 非局部均值组合；不是 AI 划痕修复，效果有限",
                            )
                            colorize_cb = gr.Checkbox(
                                value=False, label="黑白照片上色",
                                info="需要自行安装并注册上色引擎（未注册时此开关不生效）",
                                visible=enhance_extras.colorizer_available(),
                            )
                        image_btn = gr.Button("🚀 开始超分", variant="primary")
                    with gr.Column():
                        image_out = gr.Image(label="超分结果", height=420)
                        image_info = gr.Markdown("_暂无任务信息_")
                image_btn.click(
                    handle_image,
                    inputs=[image_in, preset_dd, device_dd, model_dd, lora_dd_img, half_cb,
                            outscale_sl, tile_sl, auto_download_cb, face_cb, img_fmt_dd,
                            denoise_cb, denoise_sl, scratch_cb, colorize_cb],
                    outputs=[image_out, image_info],
                )

                gr.Markdown("---\n### 📁 批量处理（P1）")
                with gr.Row():
                    batch_files = gr.File(
                        label="上传多张图片", file_count="multiple", file_types=["image"], height=160
                    )
                    with gr.Column():
                        batch_folder = gr.Textbox(
                            label="或填写服务器上的图片文件夹路径",
                            placeholder="例如 /data/photos （留空则只用上传的文件）",
                        )
                        batch_btn = gr.Button("开始批量超分")
                with gr.Row():
                    batch_gallery = gr.Gallery(label="处理结果", columns=4, height=320)
                    batch_summary = gr.Markdown("_等待任务_")

                batch_btn.click(
                    handle_image_batch,
                    inputs=[batch_files, batch_folder, preset_dd, device_dd, model_dd, lora_dd_img,
                            half_cb, outscale_sl, tile_sl, auto_download_cb, img_fmt_dd],
                    outputs=[batch_gallery, batch_summary],
                )

            # ---------------- Tab2 视频 ---------------- #
            with gr.Tab("🎬 视频超分"):
                with gr.Row():
                    with gr.Column():
                        video_in = gr.Video(label="输入视频", height=320)
                        video_source_info = gr.Markdown("_上传视频后显示媒体信息_")
                        video_model_dd = gr.Dropdown(
                            choices=video_models, value=video_models[0], label="超分模型",
                            info="Maxine 未安装时不会出现在此列表",
                        )
                        with gr.Row():
                            lora_dd_vid = gr.Dropdown(choices=lora_items, value=NO_LORA, label="LoRA 权重")
                            video_fmt_dd = gr.Dropdown(
                                choices=list(config.VIDEO_FORMATS), value=config.DEFAULT_VIDEO_FORMAT,
                                label="输出格式",
                            )
                        with gr.Group():
                            preview_cb = gr.Checkbox(value=False, label="仅预览前 N 帧（P1，快速看效果）")
                            preview_frames_sl = gr.Slider(
                                5, 300, value=config.PREVIEW_FRAME_LIMIT, step=5, label="预览帧数",
                            )
                        with gr.Row():
                            fps_tb = gr.Number(value=None, label="输出帧率覆盖", precision=3,
                                               info="留空沿用源帧率")
                            quality_dd = gr.Dropdown(
                                choices=["恒定质量", "固定码率"], value="恒定质量", label="质量控制",
                            )
                        with gr.Row():
                            crf_sl = gr.Slider(0, 51, value=config.FFMPEG_CRF, step=1,
                                               label="CRF / CQ（越小越清晰）")
                            bitrate_tb = gr.Number(value=None, label="码率 Mbps（固定码率模式）", precision=2)
                        video_btn = gr.Button("🚀 开始超分", variant="primary")
                    with gr.Column():
                        video_out = gr.Video(label="输出视频", height=380)
                        video_info = gr.Markdown("_暂无任务信息_")
                        output_download = gr.File(label="下载输出文件")

                video_in.change(preview_video_info, inputs=video_in, outputs=video_source_info)
                video_btn.click(
                    handle_video,
                    inputs=[video_in, preset_dd, device_dd, video_model_dd, lora_dd_vid, half_cb,
                            outscale_sl, tile_sl, auto_download_cb, video_fmt_dd, preview_cb,
                            preview_frames_sl, fps_tb, quality_dd, bitrate_tb, crf_sl],
                    outputs=[video_out, video_info],
                ).then(lambda path: path, inputs=video_out, outputs=output_download)

            # ---------------- Tab3 微调（P2 可选） ---------------- #
            with gr.Tab("🧪 模型微调"):
                gr.Markdown(
                    "### ⚠️ 高级可选功能\n"
                    "本页用于训练 **RealESRGAN 卷积 LoRA**（自定义风格微调）。使用前请注意：\n"
                    "1. **显存需求远高于推理**，显存不足时请减小 patch / batch 或降低 rank；\n"
                    "2. 训练**只更新 LoRA 参数**，基础模型权重不会被覆盖，导出的 `*.pth` 只含 LoRA；\n"
                    "3. **Maxine VSR 是闭源组件，不支持微调**，本页仅适用于 RealESRGAN 链路；\n"
                    "4. 数据量少、风格强烈时容易过拟合，请留意验证预览图与 PSNR 变化。"
                )
                gr.Markdown(training_status_markdown())

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 1) 训练数据（高清 HR 原图）")
                        train_files = gr.File(
                            label="上传 HR 图片（可多选）", file_count="multiple",
                            file_types=["image"], height=120,
                        )
                        train_folder = gr.Textbox(
                            label="服务器上的图片文件夹路径（推荐，可多行）",
                            placeholder="/data/hr_photos", lines=2,
                        )
                        with gr.Row():
                            preview_deg_btn = gr.Button("🔍 预览退化效果")
                            preview_count_sl = gr.Slider(1, 8, value=4, step=1, label="预览组数")
                        degrade_gallery = gr.Gallery(
                            label="退化对比（左：原图 / 右：退化后最近邻放大）", columns=2, height=220
                        )
                        degrade_note = gr.Markdown("_建议先预览退化效果，确认强度后再开训_")
                    with gr.Column():
                        gr.Markdown("#### 2) 训练参数")
                        lora_name_tb = gr.Textbox(value="my_lora", label="LoRA 名称",
                                                  info="训练完成后写入 lora_weights/")
                        with gr.Row():
                            train_model_dd = gr.Dropdown(
                                choices=model_options, value=model_options[0],
                                label="基础模型（训练起点）",
                            )
                            train_device_dd = gr.Dropdown(
                                choices=gpu_options, value=default_device, label="训练设备"
                            )
                        with gr.Row():
                            epochs_nb = gr.Number(value=10, precision=0, label="epoch")
                            batch_nb = gr.Number(value=2, precision=0, label="batch size")
                            patch_nb = gr.Number(value=128, precision=0, label="裁剪尺寸 patch")
                        with gr.Row():
                            lr_nb = gr.Number(value=1e-4, label="学习率")
                            rank_nb = gr.Number(value=8, precision=0, label="LoRA rank")
                            alpha_nb = gr.Number(value=None, label="alpha（空 = rank）")
                        with gr.Row():
                            val_ratio_sl = gr.Slider(0, 0.3, value=0.05, step=0.01, label="验证集比例")
                            val_interval_nb = gr.Number(value=1, precision=0, label="每 N epoch 验证")
                            save_interval_nb = gr.Number(value=5, precision=0, label="每 N epoch 保存")
                        with gr.Row():
                            loss_dd = gr.Dropdown(
                                choices=["l1", "l1+perceptual"], value="l1", label="损失函数"
                            )
                            percep_sl = gr.Slider(0, 1, value=0.1, step=0.01, label="感知损失权重")
                            accum_nb = gr.Number(value=1, precision=0, label="梯度累积")
                        with gr.Row():
                            amp_cb = gr.Checkbox(value=True, label="AMP 混合精度")
                            seed_nb = gr.Number(value=0, precision=0, label="随机种子")
                            resume_dd = gr.Dropdown(
                                choices=(train_mod.default_resume_candidates() if HAS_TRAIN else ["(不续训)"]),
                                value="(不续训)", label="断点续训",
                            )

                with gr.Row():
                    train_start_btn = gr.Button("🚀 开始训练", variant="primary", interactive=HAS_TRAIN)
                    train_stop_btn = gr.Button("⏹ 停止训练", interactive=False)
                train_launch_md = gr.Markdown("_尚未启动训练_")
                with gr.Row():
                    train_status_box = gr.Markdown("_等待任务_")
                    train_log_box = gr.Markdown("```\n等待训练启动…\n```")
                with gr.Row():
                    train_val_gallery = gr.Gallery(
                        label="验证预览（LR | SR(LoRA) | HR）", columns=2, height=240
                    )
                    train_output_file = gr.File(label="导出的 LoRA 权重")

                gr.Markdown("---\n#### LoRA 权重管理")
                with gr.Row():
                    lora_manage_md = gr.Markdown(lora_files_markdown())
                with gr.Row():
                    delete_lora_dd = gr.Dropdown(choices=lora_choices()[1:], label="选择要删除的 LoRA")
                    delete_lora_btn = gr.Button("🗑 删除")
                    refresh_train_lora_btn = gr.Button("🔄 刷新列表")
                train_delete_note = gr.Markdown("")

                preview_deg_btn.click(
                    handle_preview_degradation,
                    inputs=[train_files, train_folder, patch_nb, rank_nb, preview_count_sl, seed_nb],
                    outputs=[degrade_gallery, degrade_note],
                )
                train_start_btn.click(
                    handle_start_training,
                    inputs=[train_files, train_folder, lora_name_tb, train_model_dd, train_device_dd,
                            epochs_nb, batch_nb, patch_nb, lr_nb, rank_nb, alpha_nb, val_ratio_sl,
                            val_interval_nb, save_interval_nb, loss_dd, percep_sl, accum_nb, amp_cb,
                            resume_dd, seed_nb],
                    outputs=[train_launch_md, train_log_box, train_status_box],
                ).then(
                    stream_training_status,
                    outputs=[train_status_box, train_log_box, train_val_gallery, train_output_file,
                             train_start_btn, train_stop_btn],
                )
                train_stop_btn.click(handle_stop_training, outputs=[train_delete_note])
                delete_lora_btn.click(
                    handle_delete_lora,
                    inputs=[delete_lora_dd],
                    outputs=[train_delete_note, lora_manage_md, lora_dd_img, lora_dd_vid],
                )
                refresh_train_lora_btn.click(
                    lambda: (lora_files_markdown(), gr.update(choices=lora_choices()[1:])),
                    outputs=[lora_manage_md, delete_lora_dd],
                )

            # ---------------- Tab4 状态 ---------------- #
            with gr.Tab("🩺 环境与 LoRA"):
                gr.Markdown("### 运行环境")
                env_md = gr.Markdown(environment_markdown())
                gr.Markdown("### FFmpeg")
                ffmpeg_md = gr.Markdown(ffmpeg_installer.status_text())
                with gr.Row():
                    install_ffmpeg_btn = gr.Button("⬇️ 下载 FFmpeg", variant="secondary")
                gr.Markdown(
                    "<sub>没有预装 FFmpeg 时用它一键装到项目内 `tools/ffmpeg/`："
                    "不需要管理员权限，也不会改系统 PATH。</sub>"
                )
                ffmpeg_note = gr.Markdown("")
                gr.Markdown("### 编码器")
                encoder_md = gr.Markdown(encoder_markdown())
                gr.Markdown("### 可选组件")
                gr.Markdown(maxine_mod.maxine_status())
                with gr.Row():
                    refresh_env_btn = gr.Button("刷新环境信息")
                    refresh_lora_btn = gr.Button("刷新 LoRA 列表")
                gr.Markdown("### LoRA 权重目录")
                lora_md = gr.Markdown(lora_markdown())

                install_ffmpeg_btn.click(
                    handle_install_ffmpeg,
                    outputs=[ffmpeg_note, ffmpeg_md, env_md, encoder_md],
                )
                refresh_env_btn.click(
                    lambda: (environment_markdown(), encoder_markdown(),
                             ffmpeg_installer.status_text()),
                    outputs=[env_md, encoder_md, ffmpeg_md],
                )
                refresh_lora_btn.click(
                    lambda: (lora_markdown(), refresh_lora_choices(), refresh_lora_choices()),
                    outputs=[lora_md, lora_dd_img, lora_dd_vid],
                )

        gr.Markdown(
            "> 说明：本项目仅在你自己的机器上运行推理，不上传任何文件。"
            "推理产物保存在 `output/`，临时帧与音轨保存在 `temp/` 并在任务结束后自动清理。"
        )

    return interface


# --------------------------------------------------------------------------- #
# 启动
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PhotoVideoAI-SR WebUI")
    parser.add_argument("--host", default=os.environ.get("PVSR_HOST", "127.0.0.1"), help="监听地址")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PVSR_PORT", "7860")), help="监听端口")
    parser.add_argument("--share", action="store_true", help="生成 Gradio 公网分享链接")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--keep-temp", action="store_true", help="保留临时文件（调试用）")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="允许在没有 CUDA 的环境启动（仅用于界面调试，推理会失败）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.keep_temp:
        config.KEEP_TEMP_FILES = True

    log_runtime_environment()
    cleanup_stale_temp()

    if not config.cuda_available() and not args.allow_cpu:
        print(
            "❌ 未检测到可用的 CUDA 设备，程序退出。\n"
            "本项目为 NVIDIA CUDA only，请在具备 CUDA 环境的机器上运行。\n"
            "自检命令：python -c \"import torch; print(torch.cuda.is_available())\"\n"
            "（仅想查看界面可用 --allow-cpu 启动，但推理会失败。）"
        )
        return 1
    if not config.cuda_available():
        logger.warning("以 --allow-cpu 模式启动：界面可用，但 GPU 推理会失败。")

    if not config.ffmpeg_available():
        logger.warning(ffmpeg_utils.check_ffmpeg())

    interface = build_interface()
    interface.queue(default_concurrency_limit=4)

    # 尽力打开输出目录，方便用户取用结果
    try:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    launch_kwargs = {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share,
        "inbrowser": not args.no_browser,
        "show_error": True,
    }
    if gradio_major_version() >= 6:
        launch_kwargs.update({"css": CSS, "theme": gr.themes.Soft()})

    interface.launch(**launch_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
