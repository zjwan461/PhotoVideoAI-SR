# PhotoVideoAI-SR

**本地运行的图片 / 视频 AI 超分工具**：Python + Gradio WebUI，NVIDIA CUDA 推理，
FFmpeg 负责媒体编解码。老照片修复、低分辨率视频放大、批量处理，全部在你自己的机器上完成，
不上传任何文件。

* 图片超分：RealESRGAN（支持加载自定义 LoRA 权重）
* 视频超分：RealESRGAN 逐帧链路（可微调）/ NVIDIA Maxine VSR 时序链路（可选组件，闭源）
* 批量图片、预览模式、自动下载模型、任务信息面板、输出格式可选
* 协议：MIT

> 本项目仅支持 **NVIDIA CUDA** 推理。代码与文档中不写死任何显卡型号，GPU 信息一律运行时探测。

---

## 目录

- [功能特性](#功能特性)
- [目录结构](#目录结构)
- [硬件与软件前置](#硬件与软件前置)
- [安装](#安装)
- [模型下载](#模型下载)
- [快速启动](#快速启动)
- [使用说明](#使用说明)
- [参数说明](#参数说明)
- [模型对比：RealESRGAN vs Maxine](#模型对比realesrgan-vs-maxine)
- [LoRA：自定义模型风格](#lora自定义模型风格)
- [环境变量](#环境变量)
- [已知限制](#已知限制)
- [常见问题](#常见问题)
- [开发与自检](#开发与自检)
- [开源协议与致谢](#开源协议与致谢)

---

## 功能特性

| 分类 | 能力 |
| --- | --- |
| 📷 照片超分 | 单图超分；结果自动 Lanczos 缩放到 720p / 1080p / 2K / 4K / 8K 或保持原始比例；PNG / JPG 输出 |
| 📁 批量处理 | 一次上传多张图片，或指定服务器上的文件夹，结果统一写入 `output/batch/`，带成功/失败汇总 |
| 🎬 视频超分 | RealESRGAN 逐帧链路；提取并回填音轨；MP4 / MKV 输出；可覆盖输出帧率 |
| ⚡ 预览模式 | 只渲染前 N 帧（默认 30 帧）即可预览效果，不必等整片跑完 |
| 🧩 LoRA 加载 | 下拉选择 `lora_weights/` 中的权重，注入到卷积层，随时切换或卸载；不选则完全走原生基础模型 |
| 🧪 LoRA 微调 | 独立「模型微调」Tab：自动退化生成 LR、只训练 LoRA 参数、训练中出验证预览图 + PSNR、断点续训、只导出 LoRA |
| 🧹 附加增强 | 老照片可选去噪 / 轻量去划痕（传统算法，无需额外权重）；人脸修复 (GFPGAN) 与黑白上色为可插拔可选组件 |
| 🖥️ 硬件编码 | 合成视频优先使用 `h264_nvenc`，不可用时自动回退 `libx264` |
| 🩺 状态面板 | 运行时展示环境摘要、编码器可用性、可选组件状态、LoRA 列表 |
| 🛡️ 健壮性 | 显存预判与友好提示、FFmpeg 缺失提示、单帧失败降级、临时文件自动清理、输入文件只读 |

---

## 目录结构

```
PhotoVideoAI-SR/
├── app.py                          # Gradio WebUI 主入口
├── config.py                       # 全局配置、路径、预设、GPU 探测
├── requirements.txt                # 依赖（区分核心 / 可选 / 开发）
├── README.md  CONTRIBUTING.md  LICENSE  .gitignore
├── src/
│   ├── common.py                   # CUDA/显存检测、预判、清理、LoRA 扫描、日志
│   ├── ffmpeg_utils.py             # FFmpeg/FFprobe 封装：拆帧、提音频、合成、探测
│   ├── img_sr.py                   # RealESRGAN 图片超分（含 LoRA 注入接口）
│   ├── lora_runtime.py             # 卷积 LoRA 注入/加载/保存（手写实现，不依赖 peft）
│   ├── video_sr_realesrgan.py      # RealESRGAN 逐帧视频推理
│   ├── video_sr_maxine.py          # Maxine VSR 时序视频推理（可选，自动 disable）
│   ├── enhance_extras.py           # 附加增强：去噪、轻量去划痕、可插拔上色
│   └── train/                      # 【可选高级模块】LoRA 微调，独立目录
│       ├── lora_network.py         # 手写 LoRA 卷积层 / 只训练 LoRA / 权重合并
│       ├── lora_utils.py           # 注入、只导出 LoRA、推理时加载/卸载
│       ├── dataset.py              # HR 读取、增强、自动退化生成 LR、坏图过滤
│       ├── validator.py            # 训练中验证：三联预览图 + PSNR
│       └── trainer.py              # 训练循环、AMP、断点续训、后台运行器
├── tests/
│   ├── selftest.py                 # 基础自测（无 GPU 也能跑媒体链路部分）
│   └── selftest_train.py           # 微调自测（合成数据 + 微型网络，CPU 可跑）
├── temp/                           # 临时帧与音轨（任务结束自动清理）
├── output/                         # 推理输出（batch/ 存放批量结果）
├── lora_weights/                   # LoRA 权重 *.pth / *.safetensors
├── model_weights/                  # 基础模型权重，如 RealESRGAN_x4plus.pth
└── logs/                           # 运行日志（每次启动一个带时间戳的文件）
```

---

## 硬件与软件前置

**必需**

* 支持 CUDA 的 NVIDIA 显卡（`torch.cuda.is_available()` 为 True；程序会在启动时自检并给出提示）
* 与显卡驱动匹配的 NVIDIA 驱动
* Python 3.10 或更高版本
* FFmpeg / FFprobe（需能被 PATH 找到，或通过环境变量指定）

**建议**

* 显存：图片超分 ≥ 4 GB 可运行，视频逐帧超分与 4K 输出建议 ≥ 8 GB；
  程序会根据空闲显存自动选择分块大小（tile），显存紧张时可手动调小。
* 内存：批量处理与 4K 视频建议 ≥ 16 GB。
* 磁盘：视频逐帧处理会在 `temp/` 生成 PNG 帧序列，长视频请预留充足空间。

**可选**

* `nvvfx`（NVIDIA Maxine VSR）：时序视频超分。未安装时 UI 自动隐藏该选项，主流程不受影响。
* `gfpgan`：老照片人脸修复。未安装时对应开关隐藏，需要额外的 `GFPGANv1.4.pth` 权重。

> Windows 与 Linux 均可运行；Windows 用户请确认 `ffmpeg.exe` / `ffprobe.exe` 在 PATH 中。

---

## 安装

### 1. 获取代码

```bash
git clone <your-repo-url>
cd PhotoVideoAI-SR
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 3. 安装 CUDA 版 PyTorch（关键步骤）

请到 [PyTorch 官网](https://pytorch.org/get-started/locally/) 按你的驱动/CUDA 版本选择命令，例如：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

安装后自检：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

`True` 才说明 CUDA 可用。若为 `False`，请先解决环境问题再继续。

### 4. 安装项目依赖

```bash
pip install -r requirements.txt
```

### 5. 安装 FFmpeg

* Ubuntu / Debian：`sudo apt install ffmpeg`
* Windows：下载官方构建包，把 `bin` 目录加入 PATH（或设置 `PVSR_FFMPEG`）
* 验证：`ffmpeg -version` 与 `ffprobe -version` 都能输出

### 6. 安装可选依赖（按需）

```bash
pip install gfpgan          # 人脸修复
pip install safetensors     # 读取 .safetensors 格式的 LoRA
# Maxine VSR：按 NVIDIA 官方文档安装 nvvfx 相关组件
```

---

## 模型下载

程序默认使用 `model_weights/RealESRGAN_x4plus.pth`。

**方式一：自动下载（默认开启）**

界面中「自动下载缺失的基础模型」保持勾选即可，首次推理会从官方地址下载到 `model_weights/`。
若网络不可达，请改用方式二。

**方式二：手动下载**

下载后按表中的文件名放入 `model_weights/`：

| 文件名 | 说明 | 官方地址 |
| --- | --- | --- |
| `RealESRGAN_x4plus.pth` | 通用 4 倍放大（默认） | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth |
| `RealESRGAN_x4plus_anime_6B.pth` | 动漫插画向 4 倍（更小更快） | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth |
| `RealESRGAN_x2plus.pth` | 通用 2 倍 | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth |
| `GFPGANv1.4.pth`（可选） | 人脸修复权重 | https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth |

想新增自定义模型时，在 `config.MODEL_REGISTRY` 里补一条名称/地址/说明即可，UI 下拉框会自动出现。

---

## 快速启动

```bash
python app.py
```

默认监听 `http://127.0.0.1:7860`，浏览器会自动打开。

常用参数：

```bash
python app.py --host 0.0.0.0 --port 8000     # 局域网访问
python app.py --share                        # 生成 Gradio 公网链接（注意隐私）
python app.py --no-browser                   # 不自动开浏览器
python app.py --keep-temp                    # 保留 temp/ 内容，排查问题用
python app.py --allow-cpu                    # 无 CUDA 时也启动界面（仅调试，推理会失败）
```

---

## 使用说明

### 📷 照片超分

1. 上传图片（支持透明通道）。
2. 选择「输出分辨率预设」：想要极限放大就选「原始比例」，想要固定长边就选 1080p / 2K / 4K 等。
3. 需要自定义风格时，在「LoRA 权重」里选择 `lora_weights/` 中的文件；不需要就保持「不使用 LoRA」。
4. 点击 **开始超分**：右侧显示结果与任务信息（耗时、输入/输出分辨率、模型、设备、显存）。
5. 结果同时会保存到 `output/`。

老照片建议：勾选「人脸修复 (GFPGAN)」，并把超分倍率保持在 4；
「🧹 附加增强」里的**去噪**与**轻量去划痕**在超分前处理输入（传统算法，不需要额外权重，
去噪强度过大会糊掉细节）；**黑白上色**需要你先安装并注册上色引擎，否则该开关不会出现。

### 📁 批量处理

在照片 Tab 下方：上传多张图片，**或者**填写服务器上的文件夹路径（会递归扫描常见图片格式），
点击「开始批量超分」。结果输出到 `output/batch/`，界面给出成功/失败明细。

### 🎬 视频超分

1. 上传视频，界面会立刻显示源信息（分辨率、帧率、时长、帧数、编码、有无音轨）。
2. 选择「超分模型」：`RealESRGAN` 或（已安装时）`Maxine VSR`。
3. 想先看效果，就打开「仅预览前 N 帧」，默认 30 帧。
4. 需要严格对齐时长或做变速时，可填「输出帧率覆盖」；否则沿用源帧率。
5. 质量控制：`恒定质量` 用 CRF/CQ（越小越清晰），`固定码率` 用 Mbps。
6. 点击 **开始超分**：带进度条，完成后可在线预览并下载输出文件。音轨会被保留（源视频有声时）。

> **Maxine VSR 不支持 LoRA**：它是闭源组件，模型参数不可训练；需要风格定制请使用 RealESRGAN 链路。

### 🧪 模型微调

训练自定义 LoRA 的完整流程（数据准备 → 预览退化 → 开训 → 看验证图 → 导出 → 选用）
见下方 [LoRA 章节的「微调」小节](#微调训练自己的-lorawebui-模型微调tab)。

### 🩺 环境与 LoRA

查看运行环境摘要、编码器可用性、可选组件状态与 LoRA 列表，并支持一键刷新（刷新后下拉框同步更新，
新增 LoRA 文件无需重启程序）。

---

## 参数说明

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| CUDA 设备 | 自动探测 | 多卡机器可选具体设备；显示名称与显存为运行时探测结果 |
| 输出分辨率预设 | 原始比例 | 超分后按长边做 Lanczos 缩放；「原始比例」不缩放 |
| 基础模型 | `RealESRGAN_x4plus.pth` | 来自 `model_weights/` 的可用文件 |
| 半精度推理 (FP16) | 开 | 省显存、更快；出现花屏/异常可关闭 |
| 分块大小 tile | 0（自动） | 0 表示按空闲显存自动选择，显存紧张时手动调小（如 200/300） |
| 超分倍率 | 4.0 | 基础模型固有倍率；随后再缩放到目标分辨率 |
| LoRA 权重 | 不使用 | 卷积 LoRA，可随时切换；不影响未选择时的原生效果 |
| 输出格式 | PNG / MP4 | 图片 PNG、JPG；视频 MP4、MKV（MKV 下音频直接 copy） |
| 预览帧数 | 30 | 仅预览模式下生效 |
| CRF / CQ | 17 | 恒定质量模式；数值越小越清晰、文件越大 |
| 码率 Mbps | 空 | 固定码率模式 |

显存预判：任务开始前会用 `torch.cuda.mem_get_info()` 读取空闲显存并预留安全余量，
不足时在界面给出可操作的提示（关闭占显存程序、调小 tile、改用预览模式等），而不是直接抛 CUDA 堆栈。

---

## 模型对比：RealESRGAN vs Maxine

| 维度 | RealESRGAN（逐帧） | Maxine VSR（时序） |
| --- | --- | --- |
| 类型 | 开源超分网络，逐帧独立推理 | NVIDIA 闭源 SDK，按时序增强 |
| 可微调 | ✅ 支持训练并加载 LoRA 权重（见「微调」章节） | ❌ 不支持，参数不可见 |
| 风格 | 细节锐利、纹理重建强，可针对素材微调 | 画面稳定、时序一致性好，风格固定 |
| 输入 | 图片与视频通用，逐帧处理 | 视频专用，需按时间顺序喂帧 |
| 依赖 | `realesrgan` / `basicsr` | `nvvfx`（可选，缺失自动隐藏） |
| 适用场景 | 老照片、动漫、需要定制风格的素材 | 实时/准实时观看、追求画面稳定的视频 |

选择建议：**要定制风格或处理图片 → RealESRGAN；只要视频更清晰稳定、且环境具备 nvvfx → Maxine。**

---

## LoRA：自定义模型风格

本项目的 LoRA 是**卷积层低秩增量**，手写实现、不依赖 `peft`：

```
y = Conv_base(x) + (alpha / rank) * B(A(x))
```

* `A`：1×1 卷积（in → rank）；`B`：与原卷积同核的卷积（rank → out）；
* 初始化时 `B = 0`，所以**注入的瞬间输出与原始模型完全一致**，加载权重后才产生风格变化；
* 只保存 LoRA 参数、rank / alpha 与元信息，**不含基础大模型**，文件大小随 rank 与注入层数变化
  （实测：rank 4 ≈ 2.8 MB，rank 8 ≈ 5.4 MB）；
* 基础模型权重全程冻结，导出 LoRA 不会覆盖或修改 `model_weights/` 中的文件。

**用法**

1. 把 `*.pth`（或 `*.safetensors`，需安装 `safetensors`）放进 `lora_weights/`；
2. 在照片 / 视频 Tab 的「LoRA 权重」下拉框中选择；
3. 想恢复原生效果，选回「不使用 LoRA」——引擎会重新加载干净的基础模型。

**文件格式**

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

### 微调：训练自己的 LoRA（WebUI「🧪 模型微调」Tab）

> 状态：**推理加载与训练都已实现**。训练端在 `src/train/`，是独立可选模块；
> 不安装/不使用它时，图片与视频超分完全不受影响。

**原理**

冻结 RealESRGAN 主干（含基础权重），只训练 LoRA 的 `A`/`B` 参数，损失默认 `L1`
（可选 `L1 + 感知损失(lpips)`，需 `pip install lpips`）。因为只有 LoRA 参与更新，
显存占用和训练时间都远小于全量微调，导出的文件也只有几 MB。

**数据集准备**

1. 准备**高清原图**（HR），不需要自己造低清图；
2. 放进一个文件夹，在微调 Tab 填写该路径（多行/多个路径都支持），或直接上传若干张图；
3. 程序会自动退化生成 LR：随机缩放 → 高斯模糊 → sinc 振铃 → 二次退化（模糊/缩放/加噪）→
   JPEG 压缩 → 加噪/灰度，模拟真实低质来源；
4. 建议同风格 200~2000 张起步；**先点「🔍 预览退化效果」**确认退化强度合适再开训；
5. 过小或损坏的图片会被自动过滤，日志里会列出原因。

**训练参数（微调 Tab）**

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| LoRA 名称 | my_lora | 导出文件名；重名时自动加序号，不会覆盖已有权重 |
| 基础模型（训练起点） | `RealESRGAN_x4plus.pth` | 结构必须与推理时一致（例如动漫版要用 `*_anime_6B.pth`） |
| epoch / batch size / patch | 10 / 2 / 128 | 显存不足时优先减小 patch，其次 batch |
| 学习率 / rank / alpha | 1e-4 / 8 / 空(=rank) | rank 越大容量越强、越容易过拟合、文件越大 |
| 验证集比例 / 每 N epoch 验证 | 0.05 / 1 | 验证集从数据里自动切分，用固定种子退化，跨 epoch 可比 |
| 每 N epoch 保存 | 5 | 另存 `*_epochNNNN.pth`，只保留最近若干个，避免占满磁盘 |
| 损失函数 / 感知权重 | l1 / 0.1 | 未安装 lpips 时自动回退纯 L1 并在日志提示 |
| 梯度累积 / AMP | 1 / 开 | 显存紧张时用累积换 batch 效果；AMP 需 CUDA |
| 随机种子 | 0 | 固定后可复现 |
| 断点续训 | (不续训) | 选 `*.state.pt` 即可从上次 epoch、优化器状态继续 |

**训练过程与产物**

* 界面实时显示：epoch/step 进度、loss、最佳 PSNR、已用时、预计剩余、空闲显存、滚动日志；
* 每 N 个 epoch 生成验证三联图 `LR | SR(LoRA) | HR`，保存到 `logs/validate/<LoRA名>/`，
  界面上直接可看（OpenCV 不支持中文字形，图中标签用英文）；
* LoRA 权重导出到 `lora_weights/<名称>.pth`；续训状态 `*.state.pt` 放在
  `logs/train/<LoRA名>/`，**不会**混进可部署权重目录，也不会出现在推理下拉框里；
* 训练摘要（参数、loss 曲线、最佳 PSNR）写入 `logs/train/<LoRA名>/summary.json`；
* 训练可以随时「⏹ 停止」，会在当前 step 结束后保存再退出。

**过拟合提示**

LoRA 收敛很快。出现下面情况就该停：loss 还在降但验证图开始变“假”、纹理重复、
或 PSNR 由升转降。处理方式：降低 rank、增加数据多样性、减少 epoch、
或把学习率调小（例如 5e-5）。

**硬件提示**

LoRA 训练显存需求高于推理（界面会先做一次显存预判）。不足时依次尝试：
减小 patch 尺寸 → 减小 batch size → 提高梯度累积 → 降低 rank → 关闭感知损失。
`patch=128 / batch=2 / rank=8` 通常十几 GB 显存可以起步，具体取决于图片内容与基础模型。

**不要指望**用几十张图就复刻某个具体艺术家的全部风格——LoRA 更适合“方向性微调”
（更锐利、更干净、更贴近某类素材）。

**命令行方式**（不经过 WebUI）：

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

## 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `PVSR_FFMPEG` / `PVSR_FFPROBE` | `ffmpeg` / `ffprobe` | 可执行文件路径 |
| `PVSR_FFMPEG_TIMEOUT` | 3600 | 单条 FFmpeg 命令超时（秒） |
| `PVSR_TEMP_DIR` / `PVSR_OUTPUT_DIR` / `PVSR_LORA_DIR` / `PVSR_MODEL_DIR` / `PVSR_LOG_DIR` | 项目内目录 | 自定义各目录位置 |
| `PVSR_HALF` | 1 | 默认是否半精度推理 |
| `PVSR_KEEP_TEMP` | 0 | 1 = 保留临时文件 |
| `PVSR_TILE` / `PVSR_TILE_PAD` | 0 / 10 | 分块大小与边距 |
| `PVSR_VRAM_MARGIN` | 1.0 | 显存预判预留余量（GB） |
| `PVSR_PREVIEW_FRAMES` | 30 | 预览模式默认帧数 |
| `PVSR_CRF` / `PVSR_X264_PRESET` | 17 / medium | 软件编码质量参数 |
| `PVSR_HOST` / `PVSR_PORT` | 127.0.0.1 / 7860 | WebUI 监听地址与端口 |

---

## 已知限制

1. **仅支持 NVIDIA CUDA**：AMD / Apple Silicon / 纯 CPU 不在支持范围内；缺失 CUDA 时程序直接退出并提示。
2. **Maxine 冒烟依赖官方 SDK**：未安装 `nvvfx` 时该链路不可用（UI 自动隐藏）；不同版本 API 存在差异，
   程序做了兼容性检查，若你的版本报“会话创建失败”，请附版本信息提 issue。
3. **视频逐帧超分耗时与磁盘占用随帧数线性增长**：长视频建议先用预览模式确认参数。
4. **音轨处理**：源视频无音轨时输出为无声视频；MKV 下音频直接 `copy`，若源音频编码不被容器支持会失败。
5. **LoRA 需要与基础模型结构匹配**：`rank` 从权重文件自动推断；匹配度低于 50% 时会拒绝加载并说明原因，
   而不是静默地只生效一部分。
6. **黑白上色需要自行接入引擎**：本项目不内置彩色化模型权重，装好引擎后用
   `src.enhance_extras.register_colorizer(fn, "引擎名")` 注册，UI 才会出现上色开关。
7. **临时文件清理**：正常结束会清空 `temp/`；进程被强杀时可能残留，下次启动会自动清理超过 24 小时的内容。
8. **GFPGAN 为人脸专用**：非人脸图片开启该开关基本无变化，且会额外占用显存。

---

## 常见问题

**Q：启动就退出，提示未检测到 CUDA 设备。**
A：先跑 `python -c "import torch; print(torch.cuda.is_available())"`。为 `False` 通常是装了 CPU 版 PyTorch，
或驱动版本低于该 PyTorch 的要求。请重装 CUDA 版 PyTorch 或更新驱动。

**Q：报 `CUDA out of memory`。**
A：关闭其它占显存的程序；把 tile 调小（如 200）；关闭半精度或降低输出分辨率；视频任务先开预览模式。

**Q：提示找不到 FFmpeg。**
A：安装 FFmpeg 并确保 `ffmpeg`、`ffprobe` 在 PATH 中，或设置 `PVSR_FFMPEG` / `PVSR_FFPROBE` 指向绝对路径；
界面的「环境与 LoRA」Tab 可以查看实时状态。

**Q：`basicsr` 安装或导入报错（如 `torchvision.transforms.functional_tensor`）。**
A：`basicsr 1.4.2` 对较新的 torchvision 有兼容问题，常见做法是安装匹配的 torchvision 版本，
或在虚拟环境中把该导入改为 `torchvision.transforms.functional`（环境层面的临时修补，不改动本项目代码）。

**Q：输出的视频没有声音。**
A：确认源视频有音轨（视频 Tab 的源信息会显示“音频: 有/无”）；MKV 容器下音频为直接复制，若编码不被支持会失败。

**Q：怎么用上自己训练的 LoRA？**
A：直接用「🧪 模型微调」Tab 训练，完成后权重会导出到 `lora_weights/`，在下拉框里刷新即可选用，无需重启。
也可以把自己手上的同格式 LoRA（`format: photovideoai-sr-lora`）直接放进该目录。

**Q：训练报显存不足怎么办？**
A：依次尝试：减小 patch（如 96 或 64）→ 减小 batch size → 提高梯度累积 → 降低 rank →
关闭感知损失。界面在开训前会给出显存预判与具体建议。

**Q：微调 Tab 显示不可用。**
A：说明没装 PyTorch（训练需要它）。安装 CUDA 版 PyTorch 后重启即可；推理功能不受影响。

**Q：验证图里的中文显示成方块。**
A：OpenCV 的 `putText` 不支持中文字形，因此预览图标签使用英文（`LR input` / `SR (LoRA)` / `HR target`），
不影响对比效果。

---

## 开发与自检

```bash
# 语法检查
python -m compileall app.py config.py src tests

# 导入检查（不触发模型加载）
python -c "import config, src.common, src.ffmpeg_utils, src.img_sr, src.lora_runtime"

# 一键自测：配置 / 媒体链路 / 显存与推理（无 CUDA 时自动跳过推理部分）
python tests/selftest.py
# 想在没有显卡的机器上慢速验证推理逻辑（需要已放置基础模型权重）：
PVSR_SELFTEST_CPU=1 python tests/selftest.py

# 微调模块自测：合成数据 + 微型网络，CPU 即可跑完整训练/验证/续训链路
python tests/selftest_train.py

# 环境自检
python -c "import config; print(config.runtime_summary())"

# 媒体链路自检（需要 ffmpeg）
python -c "from src.ffmpeg_utils import probe_encoder_support; print(probe_encoder_support())"
```

测试清单（提交前建议覆盖）：

* 图片：普通照片放大 / 老照片压缩图放大 / 加载自定义 LoRA 对比效果
* 视频：低分辨率短视频逐帧超分 / Maxine 链路（已装 nvvfx）/ 预览模式 / 音轨保留与播放
* 边界：显存不足 / 损坏图片或视频 / FFmpeg 未安装 / 无 CUDA 设备

---

## 开源协议与致谢

本项目使用 [MIT License](LICENSE)。

* 超分模型与算法来自 [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)（BSD-3-Clause），
  请遵循其原始许可与使用条款。
* 可选人脸修复来自 [GFPGAN](https://github.com/TencentARC/GFPGAN)。
* 可选时序超分来自 NVIDIA Maxine 系列 SDK，闭源，使用需遵循 NVIDIA 官方条款。
* 媒体编解码由 [FFmpeg](https://ffmpeg.org/) 完成；WebUI 由 [Gradio](https://gradio.app/) 提供。

使用本工具处理他人作品时，请自行确认授权与合规性。
