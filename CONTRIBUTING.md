# 贡献指南

感谢你愿意为 PhotoVideoAI-SR 出力！本文档说明如何提 issue、如何改代码、以及本项目
必须遵守的几条硬性约定。

## 一、行为准则

* 友善、具体、对事不对人。技术分歧用可复现的实验与数据说话。
* 不提交任何来源不明或授权不清的模型权重、训练数据、演示素材。

## 二、环境准备

```bash
git clone https://github.com/zjwan461/PhotoVideoAI-SR.git   # 或你 fork 的地址
cd PhotoVideoAI-SR
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
python scripts/install.py                             # 一键装齐（含 CUDA 版 PyTorch）
python app.py
```

手工安装见 README「安装」章节第 4 步；国内网络可用 `python scripts/install.py --mirror cn`。

* 只支持 NVIDIA CUDA 推理；没有可用 CUDA 时程序会直接退出并提示（可用 `--allow-cpu`
  仅用于界面调试，推理必然失败）。
* FFmpeg / FFprobe 需在 PATH 中，或用 `PVSR_FFMPEG` / `PVSR_FFPROBE` 指定绝对路径。

## 三、项目约定（提交前请自查）

1. **不出现任何特定显卡型号**。所有硬件信息必须通过 `torch.cuda.get_device_properties()`
   等接口运行时探测，代码、注释、文档、日志、UI 文案一致遵守。
2. **不污染主推理链路**。LoRA 等高级能力必须可选：不启用时模型结构与原始 RealESRGAN
   完全一致；可选依赖（Maxine、GFPGAN、safetensors）缺失时必须自动降级或隐藏，不能导致
   程序无法启动。
3. **所有 IO 与外部进程调用都要有异常处理**，并把异常转换成用户能看懂的中文提示
   （参考 `src/common.py:friendly_cuda_error`、`src/ffmpeg_utils.py:FFmpegError`）。
4. **临时文件必须清理**。任何产生中间文件的流程都要放在 `temp/` 下的任务子目录，
   用 `try/finally` 保证成功或失败后都调用清理逻辑（`src.common.safe_remove`）。
5. **不修改用户原始输入文件**，输出一律写到 `output/`。
6. **GPU 推理代码必须能与 CPU 环境共存导入**：重依赖（torch / realesrgan / nvvfx / gfpgan）
   使用惰性导入或 `try: import ... except: HAS_XXX = False` 模式，便于测试与静态检查。
7. **显存相关报错要友好化**，不要把 CUDA 原始堆栈直接抛给用户。

## 四、代码风格

* Python 3.10+ 语法，`from __future__ import annotations`。
* 使用类型注解；公共函数写 docstring（中文即可），说明参数、返回值与异常。
* 单文件职责单一，UI 逻辑只放在 `app.py`，业务逻辑放在 `src/`。
* 提交前至少执行：

```bash
python -m compileall app.py config.py src tests scripts
python -c "import config, src.common, src.ffmpeg_utils, src.img_sr"
python tests/selftest.py            # 基础链路自测
python tests/selftest_train.py      # 微调链路自测（CPU 可跑）
python tests/selftest_e2e.py        # 端到端真机验收（需 CUDA + 权重；缺条件会自动跳过）
```

## 五、提交 PR 的流程

1. Fork 仓库并从 `main` 切出特性分支：`git checkout -b feat/your-feature`。
2. 一个 PR 只做一件事；涉及 UI 的改动请附截图。
3. PR 描述里写清：动机、改动点、验证方式（跑了哪些命令、什么输入输出）。
4. 涉及行为变更的，请同步更新 README 与 `app.py` 里的说明文案。

## 六、报告 Bug 时请附上

* 操作系统版本、Python 版本、`pip list` 中的 torch / realesrgan / gradio 版本。
* `nvidia-smi` 的驱动与 CUDA 版本（**不要**在 issue 中做无意义的硬件性能攀比，但版本信息必需）。
* `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` 输出。
* `logs/` 目录中对应任务的日志片段，以及能复现问题的最小输入文件。
* 完整报错信息（含堆栈）。

## 七、路线图

* P0（已完成）：基础设施、图片超分、FFmpeg 工具、逐帧视频超分、Gradio 界面与异常处理。
* P1（已完成）：预览模式、批量处理、模型自动下载、任务信息面板、FFmpeg 容错、输出格式选择。
* P2（已完成）：RealESRGAN 卷积 LoRA 微调 —— 手写 LoRA（`src/train/lora_network.py`）、
  数据集退化管线（`dataset.py`）、训练循环与断点续训（`trainer.py`）、训练中验证预览（`validator.py`）、
  WebUI「🧪 模型微调」Tab。欢迎继续改进：多卡训练、更丰富的退化算子、导出合并权重等。
* P3（已完成）：真机可用性修复与端到端验收。已修掉四类「导入没问题、一点按钮就崩」的缺陷：
  torchvision ≥ 0.17 删除 `transforms.functional_tensor` 导致 `import realesrgan` 失败
  （`src/torchvision_compat.py`）、FFmpeg 9 移除 `-vsync`（改为按版本探测 `-fps_mode`）、
  LoRA 与 FP16 主干的 dtype 不一致、UI 设备下拉标签被原样传给 torch
  （`config.normalize_device`）；并把视频任务的临时目录清理改为只清自己的子目录（并发安全）。
  对应验收脚本：`tests/selftest_e2e.py`。
