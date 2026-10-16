"""【可选高级模块】RealESRGAN LoRA 微调。

本目录是**独立的高级功能**，不参与主推理链路：不装 PyTorch 训练相关依赖、
或者完全不使用微调功能时，图片/视频超分照常工作。

模块划分
--------
* :mod:`src.train.lora_network` —— 手写 LoRA 卷积层（不依赖 peft）、只训练 LoRA、权重合并
* :mod:`src.train.lora_utils` —— 构建可训练模型、只导出 LoRA、推理时注入/卸载
* :mod:`src.train.dataset` —— HR 读取、随机裁剪/翻转/旋转、自动退化生成 LR、坏图过滤
* :mod:`src.train.validator` —— 训练中定期出三联预览图（LR | SR | HR）并计算 PSNR
* :mod:`src.train.trainer` —— 训练循环、AdamW、AMP、梯度累积、断点续训、进度回调、后台运行器

典型用法（脚本）::

    from src.train.trainer import TrainConfig, TrainingRunner
    cfg = TrainConfig(dataset_paths=["/data/hr"], lora_name="my_style", epochs=20)
    runner = TrainingRunner(cfg)
    runner.start()
    runner.join()

WebUI 里对应「🧪 模型微调」Tab。
"""

from __future__ import annotations

__all__ = [
    "lora_network",
    "lora_utils",
    "dataset",
    "validator",
    "trainer",
]

__version__ = "0.1.0"
