# PhotoVideoAI-SR

**语言 / Language**：[中文](README.md) · **English**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-NVIDIA-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**A locally run image / video AI upscaling tool**: Python + Gradio WebUI, NVIDIA CUDA inference,
with FFmpeg handling media encoding and decoding. Old photo restoration, low-resolution video
upscaling and batch processing all happen on your own machine — no file is ever uploaded.

* Image upscaling: RealESRGAN (supports loading custom LoRA weights)
* Video upscaling: RealESRGAN per-frame pipeline (fine-tunable) / NVIDIA Maxine VSR temporal pipeline (optional component, closed source)
* Batch images, preview mode, automatic model download, task info panel, selectable output format
* License: MIT

> This project only supports **NVIDIA CUDA** inference. No GPU model is hard-coded in the code or docs; GPU information is always probed at runtime.

**Repository**: [https://github.com/zjwan461/PhotoVideoAI-SR](https://github.com/zjwan461/PhotoVideoAI-SR)
· **Feedback**: [Issues](https://github.com/zjwan461/PhotoVideoAI-SR/issues)

---

## Table of Contents

- [Features](#features)
- [Directory Structure](#directory-structure)
- [Hardware and Software Prerequisites](#hardware-and-software-prerequisites)
- [Installation](#installation)
- [Model Download](#model-download)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Parameters](#parameters)
- [Model Comparison: RealESRGAN vs Maxine](#model-comparison-realesrgan-vs-maxine)
- [LoRA: Custom Model Styles](#lora-custom-model-styles)
- [Environment Variables](#environment-variables)
- [Known Limitations](#known-limitations)
- [FAQ](#faq)
- [Development and Self-Tests](#development-and-self-tests)
- [License and Acknowledgements](#license-and-acknowledgements)

---

## Features

| Category | Capability |
| --- | --- |
| 📷 Photo upscaling | Single-image upscaling; the result is automatically Lanczos-resized to 720p / 1080p / 2K / 4K / 8K or kept at the original ratio; PNG / JPG output |
| 📁 Batch processing | Upload several images at once, or point at a folder on the server; results all go to `output/batch/` with a success/failure summary |
| 🎬 Video upscaling | RealESRGAN per-frame pipeline; audio track extracted and muxed back; MP4 / MKV output; output frame rate can be overridden |
| ⚡ Preview mode | Render only the first N frames (30 by default) to preview the effect, so you need not wait for the whole video |
| 🧩 LoRA loading | Pick weights from `lora_weights/` in the dropdown, injected into the convolution layers, switch or unload at any time; nothing selected means the pure native base model |
| 🧪 LoRA fine-tuning | A separate 「模型微调 (Model Fine-tuning)」 Tab: automatic degradation to generate LR, trains LoRA parameters only, validation preview images + PSNR during training, resume from checkpoint, LoRA-only export |
| 🧹 Extra enhancements | Old photos can optionally get denoising / light descratching (classic algorithms, no extra weights); face restoration (GFPGAN) and B&W colorization are pluggable optional components |
| 🖥️ Hardware encoding | Composited video prefers `h264_nvenc`, falling back to `libx264` when unavailable |
| 🩺 Status panel | Shows the runtime environment summary, encoder availability, optional component status and the LoRA list |
| 🛡️ Robustness | VRAM pre-check with friendly hints, missing-FFmpeg hints, per-frame failure fallback, automatic temp-file cleanup, read-only input files |
| ⏹ Task cancellation | Image / batch / video tasks can be cancelled at any time, typically stopping within one second and cleaning up temporary files (works during model download too) |
| 🧹 VRAM management | One click unloads the model and reclaims the CUDA cache, so OOM no longer means restarting the program (refuses while a task is active) |

---

## Directory Structure

```
PhotoVideoAI-SR/
├── app.py                          # Gradio WebUI entry point
├── config.py                       # Global config, paths, presets, GPU probing
├── requirements.txt                # Dependencies (core / optional / dev)
├── README.md  CONTRIBUTING.md  LICENSE  .gitignore
├── src/
│   ├── common.py                   # CUDA/VRAM detection, pre-checks, cleanup, LoRA scanning, logging
│   ├── ffmpeg_utils.py             # FFmpeg/FFprobe wrapper: frame extraction, audio extraction, compositing, probing
│   ├── img_sr.py                   # RealESRGAN image upscaling (with LoRA injection hooks)
│   ├── lora_runtime.py             # Convolutional LoRA injection/loading/saving (hand-written, no peft)
│   ├── torchvision_compat.py       # Compatibility shim: fixes basicsr import failure on newer torchvision
│   ├── video_sr_realesrgan.py      # RealESRGAN per-frame video inference
│   ├── video_sr_maxine.py          # Maxine VSR temporal video inference (optional, auto-disabled)
│   ├── enhance_extras.py           # Extra enhancements: denoising, light descratching, pluggable colorization
│   └── train/                      # [Optional advanced module] LoRA fine-tuning, standalone directory
│       ├── lora_network.py         # Hand-written LoRA conv layers / train LoRA only / weight merging
│       ├── lora_utils.py           # Injection, LoRA-only export, load/unload at inference time
│       ├── dataset.py              # HR reading, augmentation, automatic degradation to LR, bad-image filtering
│       ├── validator.py            # In-training validation: three-panel preview image + PSNR
│       └── trainer.py              # Training loop, AMP, resume from checkpoint, background runner
├── tests/
│   ├── selftest.py                 # Basic self-test (the media pipeline parts run without a GPU)
│   ├── selftest_train.py           # Fine-tuning self-test (synthetic data + tiny network, runs on CPU)
│   └── selftest_e2e.py             # End-to-end real-machine acceptance (needs CUDA + weights; image/video/LoRA/fine-tuning/UI)
├── scripts/
│   └── install.py                  # One-click installer (handles CUDA torch / basicsr / mirrors)
├── temp/                           # Temporary frames and audio (auto-cleaned when a task ends)
├── output/                         # Inference output (batch/ holds batch results)
├── lora_weights/                   # LoRA weights *.pth / *.safetensors
├── model_weights/                  # Base model weights, e.g. RealESRGAN_x4plus.pth
├── tools/                          # Tools downloaded at runtime (FFmpeg from the UI button)
└── logs/                           # Runtime logs (one timestamped file per launch)
```

---

## Hardware and Software Prerequisites

**Required**

* An NVIDIA GPU with CUDA support (`torch.cuda.is_available()` returns True; the program self-checks at startup and gives a hint)
* An NVIDIA driver matching the GPU
* Python 3.10 or newer
* FFmpeg / FFprobe (must be findable on PATH, or specified through environment variables)

**Recommended**

* VRAM: ≥ 4 GB runs image upscaling; ≥ 8 GB is recommended for per-frame video upscaling and 4K output;
  the program picks the tile size automatically from free VRAM, and you can lower it manually when VRAM is tight.
* RAM: ≥ 16 GB is recommended for batch processing and 4K video.
* Disk: per-frame video processing writes PNG frame sequences into `temp/`, so reserve plenty of space for long videos.

**Optional**

* `nvvfx` (NVIDIA Maxine VSR): temporal video upscaling. When it is not installed the UI hides that option automatically, and the main flow is unaffected.
* `gfpgan`: face restoration for old photos. When it is not installed the corresponding toggle is hidden; it also requires the extra `GFPGANv1.4.pth` weights.

> Runs on both Windows and Linux; Windows users should make sure `ffmpeg.exe` / `ffprobe.exe` are on PATH.

---

## Installation

### 1. Get the Code

```bash
git clone https://github.com/zjwan461/PhotoVideoAI-SR.git
cd PhotoVideoAI-SR
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate
```

> **Note for Windows users**: please use the installer from [python.org](https://www.python.org/downloads/)
> or a conda environment. The Microsoft Store `python.exe` is only a placeholder stub, and installing
> dependencies will fail (the install script detects this and prints a hint).

### 3. Install Dependencies (Recommended: One-Click Script)

```bash
python scripts/install.py
```

The script handles the three most common pitfalls in the right order, and is idempotent and repeatable:

1. **PyTorch must be the CUDA build** — by default PyPI hands you CPU wheels on Windows, and after
   installation `torch.cuda.is_available()` is `False`, so the program refuses to start at all;
2. **basicsr cannot be installed directly** — the metadata of `basicsr==1.4.2` declares `tb-nightly`,
   yet that package no longer exists on PyPI (`No matching distribution found for tb-nightly`),
   so it must be installed with `--no-deps` and its runtime dependencies filled in manually;
3. **basicsr is a source distribution** — pip's isolated build environment reconnects to the default
   index to install build dependencies, which on Windows often hangs at `Installing build dependencies`
   for a dozen minutes, hence `--no-build-isolation`.

On networks in mainland China add `--mirror cn` (PyPI through Tsinghua, PyTorch wheels through the SJTU mirror; measured more than an order of magnitude faster):

```bash
python scripts/install.py --mirror cn
```

Other options:

```bash
python scripts/install.py --skip-torch   # CUDA PyTorch is already installed
python scripts/install.py --cpu          # install the CPU build only (UI only, inference will fail)
python scripts/install.py --dry-run      # only print the commands that would be executed
```

### 4. Manual Installation (Without the Script)

```bash
# 4.1 Install the CUDA build of PyTorch first (do not use the default index; on Windows it is the CPU build)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# Mirror in mainland China: --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu126

# 4.2 Core dependencies (this file installs with plain pip; it does not include torch / basicsr / realesrgan)
pip install -r requirements.txt

# 4.3 Upscaling model stack — the order must not be changed
pip install addict future lmdb pyyaml requests scikit-image scipy yapf facexlib
pip install --no-build-isolation --no-deps basicsr==1.4.2
pip install --no-deps realesrgan==0.3.0
```

Self-check after installation (the last value of the first command must be `True`, the second should print `True ''`):

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import src.img_sr as m; print(m.HAS_REALESRGAN, m.REALESRGAN_IMPORT_ERROR)"
```

### 5. Install FFmpeg

* Ubuntu / Debian: `sudo apt install ffmpeg`
* Windows: download an official build and add the `bin` directory to PATH (or set `PVSR_FFMPEG`)
* Verify: both `ffmpeg -version` and `ffprobe -version` produce output
* Version: FFmpeg 4.x ~ 9.x all work. The program probes the version and automatically picks either `-vsync 0` or
  `-fps_mode passthrough` (`-vsync` was removed in FFmpeg 9)

> **Do not want to install it by hand?** Start the WebUI, open the 「🩺 环境与 LoRA (Environment & LoRA)」 Tab
> and click **⬇️ 下载 FFmpeg (Download FFmpeg)**. The program downloads a prebuilt package into the project's
> `tools/ffmpeg/` (about 190 MB on Windows, roughly 340 MB once extracted; about 40 MB on Linux) and takes
> effect immediately — **no administrator rights, no changes to the system PATH, and no restart**.
> Download sources are tried in order, and if throughput stays below 64 KB/s for 20 seconds it switches
> to the next source. On restricted networks, point `PVSR_FFMPEG_URL` at a mirror.

### 6. Install Optional Dependencies (As Needed)

```bash
pip install gfpgan          # Face restoration: the library alone is not enough, you also
                            # need GFPGANv1.4.pth (~350 MB) in model_weights/, otherwise
                            # the UI hides the toggle
pip install lpips           # Optional perceptual loss for fine-tuning (downloads VGG
                            # weights, ~500 MB, on first use)
# Maxine VSR: install the nvvfx components following NVIDIA's official documentation
```

> `safetensors` is a default dependency (it reads third-party `.safetensors` LoRA files),
> so no extra install is needed. `scikit-image` is **not** a dependency of this project;
> it comes with basicsr (see step 4.3).

---

## Model Download

The program uses `model_weights/RealESRGAN_x4plus.pth` by default.

**Option 1: Automatic download (enabled by default)**

Just keep the 「自动下载缺失的基础模型 (Auto-download missing base models)」 checkbox ticked in the UI, and the first
inference downloads the model from the official address into `model_weights/`. If the network is unreachable, use option 2 instead.

**Option 2: Manual download**

After downloading, put the files into `model_weights/` using the filenames in the table:

| File name | Description | Official URL |
| --- | --- | --- |
| `RealESRGAN_x4plus.pth` | General 4× upscaling (default) | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth |
| `RealESRGAN_x4plus_anime_6B.pth` | 4× for anime illustrations (smaller and faster) | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth |
| `RealESRGAN_x2plus.pth` | General 2× | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth |
| `GFPGANv1.4.pth` (optional) | Face restoration weights | https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth |

> **Note on networks in mainland China**: these files are hosted on GitHub Releases, where a direct
> connection often gives only a few dozen KB/s (the default model is about 64 MB). If there is no progress
> for a long time, download through a proxy and place the file into `model_weights/` by hand:
>
> ```bash
> curl -L --proxy http://127.0.0.1:1080 -o model_weights/RealESRGAN_x4plus.pth \
>   https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
> ```
>
> When the download is finished you can check whether the file is intact with `python -c "import torch;torch.load('model_weights/RealESRGAN_x4plus.pth',map_location='cpu')"`
> (a corrupt file raises a zip archive related error).

To add a custom model, just add one entry with name/URL/description to `config.MODEL_REGISTRY` and it appears in the UI dropdown automatically.

---

## Quick Start

```bash
python app.py
```

By default it listens on `http://127.0.0.1:7860` and the browser opens automatically.

Common options:

```bash
python app.py --host 0.0.0.0 --port 8000     # LAN access
python app.py --share                        # generate a public Gradio link (mind your privacy)
python app.py --no-browser                   # do not open the browser automatically
python app.py --keep-temp                    # keep temp/ contents for troubleshooting
python app.py --allow-cpu                    # start the UI even without CUDA (debugging only, inference will fail)
```

---

## Usage

### 📷 Photo Upscaling

1. Upload an image (alpha channel supported).
2. Choose the 「输出分辨率预设 (Output resolution preset)」: pick 「原始比例 (Original ratio)」 for extreme upscaling, or 1080p / 2K / 4K etc. for a fixed long edge.
3. To use a custom style, choose a file from `lora_weights/` in 「LoRA 权重 (LoRA weights)」; if you do not need one, keep 「不使用 LoRA (No LoRA)」.
4. Click **开始超分 (Start Upscaling)**: the result and task info (elapsed time, input/output resolution, model, device, VRAM) appear on the right.
5. The result is also saved to `output/`.

For old photos: tick 「人脸修复 (GFPGAN)」 (Face Restoration) and keep the upscaling factor at 4;
**去噪 (Denoise)** and **轻量去划痕 (Light Descratch)** under 「🧹 附加增强 (Extra Enhancements)」 process the input
before upscaling (classic algorithms, no extra weights needed; too strong a denoise strength blurs away detail);
**黑白上色 (B&W Colorization)** requires you to install and register a colorization engine first, otherwise that toggle does not appear.

### 📁 Batch Processing

Below the photo Tab: upload multiple images, **or** fill in a folder path on the server (common image formats are
scanned recursively), then click 「开始批量超分 (Start batch upscaling)」. Results are written to `output/batch/`, and the UI lists successes and failures in detail.

### 🎬 Video Upscaling

1. Upload a video and the UI immediately shows source info (resolution, frame rate, duration, frame count, codec, whether there is an audio track).
2. Choose the 「超分模型 (Upscaling model)」: `RealESRGAN` or, when installed, `Maxine VSR`.
3. To see the effect first, turn on 「仅预览前 N 帧 (Preview first N frames only)」, 30 frames by default.
4. If you need strict duration alignment or speed changes, fill in 「输出帧率覆盖 (Output frame rate override)」; otherwise the source frame rate is used.
5. Quality control: 「恒定质量 (Constant Quality)」 uses CRF/CQ (lower is sharper), 「固定码率 (Fixed Bitrate)」 uses Mbps.
6. Click **开始超分 (Start Upscaling)**: a progress bar is shown, and when it finishes you can preview and download the output file online. The audio track is preserved (when the source video has audio).

> **Maxine VSR does not support LoRA**: it is a closed-source component whose model parameters cannot be trained; use the RealESRGAN pipeline if you need style customization.

### 🧪 模型微调 (Model Fine-tuning)

The complete workflow for training a custom LoRA (data preparation → preview degradation → start training → check validation images → export → select)
is described in the [「Fine-tuning」 subsection of the LoRA chapter](#fine-tuning-training-your-own-lora-webui-模型微调-tab) below.

### 🩺 环境与 LoRA (Environment & LoRA)

View the runtime environment summary, encoder availability, optional component status and the LoRA list, with one-click refresh (the dropdowns update after refreshing, so newly added LoRA files need no program restart).

### ⏹ Cancel a Task / 🧹 Release VRAM

Two buttons sit permanently at the top of the page (above the global settings), available from every Tab.

**⏹ 取消当前任务 (Cancel current task)**

* click it while an image / batch / video task is running and it usually stops **within about a second**,
  leaving no temporary files behind;
* it also works while the base model is being downloaded or loaded, so you do not have to wait it out;
* images already finished in a batch stay in `output/batch/`; a cancelled video task produces no half-written file;
* cancellation is cooperative: a cross-thread token is checked at block-level forward passes and at each
  stage boundary, so there may be slightly under a second between "request sent" and the task actually stopping.

**🧹 释放显存 (Release VRAM)**

* unloads the loaded model and reclaims the CUDA cache, so a `CUDA out of memory` error **no longer requires
  restarting the program**;
* measured: after a 4K inference a single click reclaims more than 20 GB of cached memory;
* it reports which model / LoRA was unloaded and the free VRAM on cuda:0 before and after;
* it **refuses to run while a task is active** (so it never yanks the model out from under an inference) and
  tells you to cancel first;
* the next task reloads the model on demand, in about a second.

---

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| CUDA 设备 (CUDA device) | Auto-detected | On multi-GPU machines a specific device can be chosen; the displayed name and VRAM are runtime probing results |
| 输出分辨率预设 (Output resolution preset) | 原始比例 (Original ratio) | After upscaling, Lanczos-resize by long edge; 「原始比例 (Original ratio)」 does not resize |
| 基础模型 (Base model) | `RealESRGAN_x4plus.pth` | Available files from `model_weights/` |
| 半精度推理 (FP16) (Half-precision inference) | On | Saves VRAM and is faster; turn it off if artifacts or anomalies appear |
| 分块大小 tile (Tile size) | 0 (auto) | 0 means choose automatically from free VRAM; lower it manually when VRAM is tight (e.g. 200/300) |
| 超分倍率 (Upscaling factor) | 4.0 | Factor inherent to the base model; the result is then resized to the target resolution |
| LoRA 权重 (LoRA weights) | 不使用 (None) | Convolutional LoRA, switchable at any time; does not affect native results when nothing is selected |
| 输出格式 (Output format) | PNG / MP4 | Images PNG, JPG; video MP4, MKV (audio is copied directly in MKV) |
| 预览帧数 (Preview frames) | 30 | Only effective in preview mode |
| CRF / CQ | 17 | Constant-quality mode; the smaller the value, the sharper the image and the larger the file |
| 码率 Mbps (Bitrate Mbps) | Empty | Fixed-bitrate mode |

VRAM pre-check: before a task starts, `torch.cuda.mem_get_info()` is used to read free VRAM and reserve a safety margin;
when it is insufficient the UI gives actionable hints (close VRAM-hungry programs, lower the tile size, switch to preview mode, etc.)
instead of throwing a raw CUDA stack trace.

---

## Model Comparison: RealESRGAN vs Maxine

| Dimension | RealESRGAN (per-frame) | Maxine VSR (temporal) |
| --- | --- | --- |
| Type | Open-source upscaling network, independent per-frame inference | NVIDIA closed-source SDK, enhances temporally |
| Fine-tunable | ✅ Supports training and loading LoRA weights (see the 「Fine-tuning」 chapter) | ❌ Not supported, parameters not exposed |
| Style | Sharp detail, strong texture reconstruction, fine-tunable for specific footage | Stable picture, good temporal consistency, fixed style |
| Input | Works for both images and video, processed frame by frame | Video only, frames must be fed in temporal order |
| Dependency | `realesrgan` / `basicsr` | `nvvfx` (optional, hidden automatically when missing) |
| Use cases | Old photos, anime, footage that needs a custom style | Real-time / near-real-time viewing, videos where a stable picture matters |

Recommendation: **custom style or image processing → RealESRGAN; just want sharper, more stable video and nvvfx is available → Maxine.**

---

## LoRA: Custom Model Styles

The LoRA in this project is a **low-rank increment on convolution layers**, hand-written and independent of `peft`:

```
y = Conv_base(x) + (alpha / rank) * B(A(x))
```

* `A`: 1×1 convolution (in → rank); `B`: convolution with the same kernel as the original convolution (rank → out);
* At initialization `B = 0`, so **the output at the moment of injection is exactly identical to the original model**; style changes only appear after weights are loaded;
* Only LoRA parameters, rank / alpha and metadata are saved, **the large base model is not included**; the file size varies with rank and the number of injected layers
  (measured: rank 4 ≈ 2.8 MB, rank 8 ≈ 5.4 MB);
* Base model weights stay frozen throughout, and exporting a LoRA never overwrites or modifies files in `model_weights/`.

**Usage**

1. Put `*.pth` or `*.safetensors` files into `lora_weights/` (both can be read; this project always exports `.pth`);
2. Select it in the 「LoRA 权重 (LoRA weights)」 dropdown of the photo / video Tab;
3. To restore the native effect, select 「不使用 LoRA (No LoRA)」 again — the engine reloads a clean base model.

**File format**

```python
{
    "format": "photovideoai-sr-lora",
    "format_version": 1,
    "rank": 8,
    "alpha": 8.0,
    "base_model": "RealESRGAN_x4plus.pth",
    "meta": {...},
    "lora_state_dict": { "body.0.rdb1.conv1.lora_A.weight": ..., ... },
}
```

### Fine-Tuning: Training Your Own LoRA (WebUI "模型微调" Tab)

> Status: **both inference loading and training are implemented**. The training side lives in `src/train/` as a standalone optional module;
> when it is not installed or not used, image and video upscaling are completely unaffected.

**How it works**

The RealESRGAN backbone (including the base weights) is frozen and only the LoRA `A`/`B` parameters are trained; the loss defaults to `L1`
(optionally `L1 + perceptual loss (lpips)`, which requires `pip install lpips`). Because only LoRA takes part in the update,
VRAM usage and training time are far below full fine-tuning, and the exported file is only a few MB.

**Dataset preparation**

1. Prepare **high-resolution source images** (HR); you do not need to create low-resolution versions yourself;
2. Put them into a folder and fill in that path in the fine-tuning Tab (multiple lines / multiple paths are supported), or upload a few images directly;
3. The program degrades them into LR automatically: random resize → Gaussian blur → sinc ringing → second degradation (blur/resize/noise) →
   JPEG compression → noise/grayscale, simulating real low-quality sources;
4. 200–2000 images of the same style are a good starting point; **click 「🔍 预览退化效果 (Preview degradation)」 first** to confirm the degradation strength is suitable before training;
5. Images that are too small or corrupt are filtered out automatically, and the log lists the reasons.

**Training parameters (fine-tuning Tab)**

| Parameter | Default | Description |
| --- | --- | --- |
| LoRA 名称 (LoRA name) | my_lora | Export file name; a numeric suffix is added on duplicates, and existing weights are never overwritten |
| 基础模型（训练起点）(Base model (training start point)) | `RealESRGAN_x4plus.pth` | The architecture must match inference (e.g. the anime version must use `*_anime_6B.pth`) |
| epoch / batch size / patch | 10 / 2 / 128 | When VRAM is short, reduce patch first, then batch |
| 学习率 / rank / alpha (Learning rate / rank / alpha) | 1e-4 / 8 / empty (=rank) | The larger the rank, the more capacity, the easier it overfits and the larger the file |
| 验证集比例 / 每 N epoch 验证 (Validation split / validate every N epochs) | 0.05 / 1 | The validation set is split from the data automatically and degraded with a fixed seed, so results are comparable across epochs |
| 每 N epoch 保存 (Save every N epochs) | 5 | Saves `*_epochNNNN.pth` separately, keeping only the most recent few to avoid filling the disk |
| 损失函数 / 感知权重 (Loss function / perceptual weight) | l1 / 0.1 | Falls back to pure L1 with a log notice when lpips is not installed |
| 梯度累积 / AMP (Gradient accumulation / AMP) | 1 / on | Trade accumulation for batch effect when VRAM is tight; AMP requires CUDA |
| 随机种子 (Random seed) | 0 | Fix it for reproducibility |
| 断点续训 (Resume from checkpoint) | (no resume) | Choose a `*.state.pt` to continue from the last epoch and optimizer state |

**Training process and artifacts**

* Live UI: epoch/step progress, loss, best PSNR, elapsed time, estimated remaining time, free VRAM, scrolling log;
* Every N epochs a three-panel validation image `LR | SR(LoRA) | HR` is generated and saved to `logs/validate/<LoRA name>/`,
  viewable right in the UI (OpenCV has no Chinese glyphs, so the labels in the image are English);
* LoRA weights are exported to `lora_weights/<name>.pth`; resume states `*.state.pt` are kept in
  `logs/train/<LoRA name>/` and **never** mix into the deployable weights directory, nor appear in the inference dropdown;
* The training summary (parameters, loss curve, best PSNR) is written to `logs/train/<LoRA name>/summary.json`;
* Training can be stopped at any time with 「⏹ 停止 (Stop)」; it saves after the current step finishes and then exits.

**Overfitting hints**

LoRA converges quickly. Stop when the following happens: loss is still dropping but validation images start to look “fake”, textures repeat,
or PSNR turns from rising to falling. Remedies: lower the rank, add data diversity, reduce epochs,
or lower the learning rate (e.g. 5e-5).

**Hardware hints**

LoRA training needs more VRAM than inference (the UI does a VRAM pre-check first). If it is insufficient, try in order:
reduce patch size → reduce batch size → increase gradient accumulation → lower rank → disable perceptual loss.
`patch=128 / batch=2 / rank=8` usually works as a starting point on a dozen-plus GB of VRAM, depending on the image content and the base model.

**Do not expect** to replicate a specific artist's entire style with a few dozen images — LoRA is better suited to “directional fine-tuning”
(sharper, cleaner, closer to a certain kind of material).

**Command-line path** (bypassing the WebUI):

```python
from src.train.trainer import TrainConfig, TrainingRunner

cfg = TrainConfig(dataset_paths=["/data/hr_photos"], lora_name="my_style",
                  epochs=20, batch_size=2, patch_size=128, rank=8)
runner = TrainingRunner(cfg)
runner.start()
runner.join()
print(runner.status(), runner.result)
```

---

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `PVSR_FFMPEG` / `PVSR_FFPROBE` | `ffmpeg` / `ffprobe` | Executable paths (highest priority; overrides the bundled FFmpeg) |
| `PVSR_FFMPEG_URL` | Chosen per platform | Custom FFmpeg download URL (internal mirror / proxy) |
| `PVSR_TOOLS_DIR` | `tools/` inside the project | Install directory for bundled tools |
| `PVSR_FFMPEG_TIMEOUT` | 3600 | Timeout for a single FFmpeg command (seconds) |
| `PVSR_TEMP_DIR` / `PVSR_OUTPUT_DIR` / `PVSR_LORA_DIR` / `PVSR_MODEL_DIR` / `PVSR_LOG_DIR` | Directories inside the project | Customize the location of each directory |
| `PVSR_HALF` | 1 | Whether to use half-precision inference by default |
| `PVSR_KEEP_TEMP` | 0 | 1 = keep temporary files |
| `PVSR_TILE` / `PVSR_TILE_PAD` | 0 / 10 | Tile size and padding |
| `PVSR_VRAM_MARGIN` | 1.0 | Safety margin reserved by the VRAM pre-check (GB) |
| `PVSR_PREVIEW_FRAMES` | 30 | Default frame count for preview mode |
| `PVSR_CRF` / `PVSR_X264_PRESET` | 17 / medium | Software encoding quality parameters |
| `PVSR_HOST` / `PVSR_PORT` | 127.0.0.1 / 7860 | WebUI listen address and port |

---

## Known Limitations

1. **NVIDIA CUDA only**: AMD / Apple Silicon / pure CPU are out of scope; without CUDA the program exits immediately with a message.
2. **The Maxine smoke test depends on the official SDK**: without `nvvfx` installed that pipeline is unavailable (the UI hides it automatically); APIs differ between versions,
   and the program includes compatibility checks — if your version reports “session creation failed”, please attach the version information in an
   [Issues](https://github.com/zjwan461/PhotoVideoAI-SR/issues) report.
3. **Per-frame video upscaling time and disk usage grow linearly with the frame count**: for long videos, confirm the parameters in preview mode first.
4. **Audio track handling**: when the source video has no audio track the output is a silent video; in MKV audio is copied directly, which fails if the source audio codec is not supported by the container.
5. **A LoRA must match the base model architecture**: `rank` is inferred from the weight file automatically; when the match is below 50% loading is refused with an explanation,
   rather than silently applying only part of it.
6. **B&W colorization requires you to integrate an engine yourself**: this project ships no colorization model weights; after installing an engine, register it with
   `src.enhance_extras.register_colorizer(fn, "engine name")` and only then does the colorization toggle appear in the UI.
7. **Temporary file cleanup**: each video task only cleans its own subdirectory under `temp/` (concurrent tasks do not affect each other);
   if the process is force-killed, leftovers may remain, and anything older than 24 hours is cleaned automatically on the next start.
8. **GFPGAN is face-specific**: enabling the toggle on images without faces changes almost nothing and it occupies extra VRAM.
9. **Compatibility of dependency versions is handled by this project as a fallback**: the pitfall of torchvision ≥ 0.17 removing `transforms.functional_tensor`
   is handled automatically by `src/torchvision_compat.py`; the pitfall of FFmpeg 9 removing `-vsync` is handled automatically by version probing.
   numpy passes the full self-test on both **1.26.4 and 2.4.6** (including real inference and fine-tuning).

---

## FAQ

**Q: It exits on startup saying no CUDA device was detected.**
A: First run `python -c "import torch; print(torch.cuda.is_available())"`. `False` usually means a CPU build of PyTorch is installed,
or the driver is older than that PyTorch requires. Reinstall the CUDA build of PyTorch or update the driver.

**Q: It reports `CUDA out of memory`.**
A: Close other VRAM-hungry programs; lower the tile size (e.g. 200); turn off half precision or lower the output resolution; for video tasks start with preview mode.

**Q: It says FFmpeg cannot be found.**
A: The easiest fix: start the WebUI, open the 「🩺 环境与 LoRA (Environment & LoRA)」 Tab and click
**⬇️ 下载 FFmpeg (Download FFmpeg)**. The program installs it into the project's `tools/ffmpeg/` and
takes effect immediately (no administrator rights, no PATH changes).
You can also install FFmpeg yourself and make sure `ffmpeg` and `ffprobe` are on PATH, or set
`PVSR_FFMPEG` / `PVSR_FFPROBE` to absolute paths.
Priority: environment variables > bundled copy > PATH; delete `tools/ffmpeg/` to fall back to the system build.

**Q: Clicking "Download FFmpeg" fails or is very slow.**
A: The program tries several download sources in order and switches to the next one when throughput stays
below 64 KB/s for 20 seconds. If none of them works (common on restricted networks), point
`PVSR_FFMPEG_URL` at an internal mirror or a reachable URL:

```bash
set PVSR_FFMPEG_URL=https://your-mirror/ffmpeg-release-essentials.zip    # Windows
export PVSR_FFMPEG_URL=https://your-mirror/ffmpeg-release-essentials.zip # Linux
```

You can also download and unpack it by hand, then put `ffmpeg(.exe)` / `ffprobe(.exe)` into `tools/ffmpeg/`.

**Q: Importing `realesrgan` raises `No module named 'torchvision.transforms.functional_tensor'`.**
A: This is **not** a missing basicsr installation; torchvision ≥ 0.17 removed that submodule while `basicsr 1.4.2`
still imports it at module top level (through `basicsr.data.degradations`, which drags `realesrgan` in).
This project already ships a built-in compatibility shim (`src/torchvision_compat.py`, which registers a forwarding module before basicsr is imported),
so there is **no need** to patch site-packages or downgrade torchvision by hand. If it still fails, `img_sr` reports an explicit hint
including the shim installation result.

**Q: `pip install basicsr` reports `No matching distribution found for tb-nightly`.**
A: The metadata of `basicsr 1.4.2` declares `tb-nightly`, but that package is no longer available on PyPI.
Install basicsr with `--no-deps` and fill in the dependencies manually; the commands are in the [Installation](#installation) chapter,
or just run `python scripts/install.py`.
`tb-nightly` is only used for basicsr's own TensorBoard logging, and this project never calls it.

**Q: pip hangs at `Installing build dependencies` for a long time while installing dependencies.**
A: That is the basicsr source distribution reconnecting to the default PyPI inside the **isolated build environment** to install build dependencies, which takes very long on networks in mainland China.
Use `pip install --no-build-isolation --no-deps basicsr==1.4.2` instead (setuptools/wheel must be installed first).

**Q: Frame extraction reports `Unrecognized option 'vsync'`.**
A: FFmpeg 9 has removed `-vsync`. This project switches to
`-fps_mode passthrough` automatically based on `ffmpeg -version`, and falls back at runtime when version probing is inaccurate, so normally no intervention is needed.

**Q: The output video has no sound.**
A: Confirm the source video has an audio track (the source info in the video Tab shows 「音频: 有/无 (Audio: Yes/No)」); in the MKV container audio is copied directly, which fails if the codec is not supported.

**Q: How do I use a LoRA I trained myself?**
A: Train it directly with the 「🧪 模型微调 (Model Fine-tuning)」 Tab; when training finishes the weights are exported to `lora_weights/`, and you can refresh the dropdown to select it, with no restart needed.
You can also drop a LoRA of the same format you already have (`format: photovideoai-sr-lora`) straight into that directory.

**Q: What should I do if training reports insufficient VRAM?**
A: Try in order: reduce patch (e.g. 96 or 64) → reduce batch size → increase gradient accumulation → lower rank →
disable perceptual loss. Before training starts the UI gives a VRAM estimate and concrete suggestions.

**Q: The fine-tuning Tab shows as unavailable.**
A: That means PyTorch is not installed (training needs it). Install the CUDA build of PyTorch and restart; inference is unaffected.

**Q: Chinese text in validation images shows up as boxes.**
A: OpenCV's `putText` has no Chinese glyphs, so the preview image labels are in English (`LR input` / `SR (LoRA)` / `HR target`),
which does not affect the comparison.

---

## Development and Self-Tests

```bash
# Syntax check
python -m compileall app.py config.py src tests scripts

# Import check (does not trigger model loading)
python -c "import config, src.common, src.ffmpeg_utils, src.img_sr, src.lora_runtime"

# One-shot self-test: config / media pipeline / VRAM and inference (the inference part is skipped automatically without CUDA)
python tests/selftest.py
# To slowly verify the inference logic on a machine without a GPU (base model weights must already be in place):
PVSR_SELFTEST_CPU=1 python tests/selftest.py

# Fine-tuning module self-test: synthetic data + tiny network, the full train/validate/resume chain runs on CPU
python tests/selftest_train.py

# End-to-end real-machine acceptance: real weights + a real GPU, covering image 4K / LoRA / video / GPU fine-tuning / UI handlers
python tests/selftest_e2e.py                 # run everything (about 20 seconds)
python tests/selftest_e2e.py --skip-train    # skip fine-tuning, verify inference and the UI only
python tests/selftest_e2e.py --keep          # keep temporary artifacts for troubleshooting

# Environment self-check
python -c "import config; print(config.runtime_summary())"

# Media pipeline self-check (requires ffmpeg)
python -c "from src.ffmpeg_utils import probe_encoder_support; print(probe_encoder_support())"
```

Test checklist (recommended coverage before submitting):

* Images: normal photo upscaling / compressed old-photo upscaling / comparing the effect of a loaded custom LoRA
* Video: per-frame upscaling of a short low-resolution clip / Maxine pipeline (with nvvfx installed) / preview mode / audio preserved and playable
* Edge cases: insufficient VRAM / corrupt image or video / FFmpeg not installed / no CUDA device

> `tests/selftest_e2e.py` covers the main paths of “image / video / LoRA / fine-tuning” listed above,
> and skips automatically returning 0 when CUDA or weights are missing, so it is suitable for CI smoke tests on machines without a GPU.

---

## License and Acknowledgements

This project is released under the [MIT License](LICENSE).

* The upscaling models and algorithms come from [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (BSD-3-Clause);
  please follow its original license and terms of use.
* Optional face restoration comes from [GFPGAN](https://github.com/TencentARC/GFPGAN).
* Optional temporal upscaling comes from the NVIDIA Maxine family of SDKs, which are closed source; usage is subject to NVIDIA's official terms.
* Media encoding and decoding are done by [FFmpeg](https://ffmpeg.org/); the WebUI is provided by [Gradio](https://gradio.app/).

When using this tool on other people's works, please confirm authorization and compliance yourself.
