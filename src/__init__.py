"""PhotoVideoAI-SR 源码包。

模块划分：
* :mod:`src.common` —— 环境探测、显存预判、清理、日志等通用工具
* :mod:`src.ffmpeg_utils` —— FFmpeg / FFprobe 封装
* :mod:`src.img_sr` —— RealESRGAN 图片超分（含 LoRA 注入接口）
* :mod:`src.lora_runtime` —— 卷积 LoRA 注入 / 加载 / 保存（手写实现，不依赖 peft）
* :mod:`src.video_sr_realesrgan` —— 逐帧视频超分
* :mod:`src.video_sr_maxine` —— 可选时序视频超分（闭源，不支持微调）
* :mod:`src.enhance_extras` —— 附加增强：去噪、轻量去划痕、可插拔上色
* :mod:`src.train` —— 【可选高级模块】LoRA 微调（手写 LoRA / 数据集 / 训练 / 验证）
"""

__version__ = "0.1.0"
__license__ = "MIT"
