"""训练中验证：定期用当前 LoRA 生成超分预览图并计算 PSNR。

产出物
------
* 预览图：``logs/validate/<run_tag>/epoch_0007.png``，内容为
  ``LR（放大到同尺寸） | SR（当前 LoRA） | HR`` 三联对比，带英文标签
  （OpenCV 的 putText 不支持中文字形，故标签用英文，避免出现“口口口”）；
* 指标：每张验证图的 PSNR（dB），以及均值，用于判断是否过拟合/劣化。

说明：验证只做前向推理，不参与梯度；默认取前 ``max_images`` 张验证图，
并用**固定随机种子**退化，保证不同 epoch 之间可比。
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from pathlib import Path

import config
from src.common import get_logger
from src.train import dataset as ds

logger = get_logger("validator")

try:
    import cv2

    HAS_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    HAS_CV2 = False

try:
    import numpy as np

    HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]
    HAS_NUMPY = False

try:
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False


class ValidatorError(RuntimeError):
    """验证阶段的可读错误。"""


def validate_root(tag: str = "default") -> Path:
    """验证图输出根目录。"""
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(tag)) or "default"
    path = config.LOG_DIR / "validate" / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# 指标与推理
# --------------------------------------------------------------------------- #
def compute_psnr(pred_rgb, target_rgb) -> float:
    """计算两张 uint8 RGB 图的 PSNR（dB）；完全一致时返回 99.0。"""
    if not HAS_NUMPY:
        raise ValidatorError("计算 PSNR 需要 numpy。")
    pred = pred_rgb.astype(np.float32)
    target = target_rgb.astype(np.float32)
    if pred.shape != target.shape:
        raise ValidatorError(f"尺寸不一致，无法比较：{pred.shape} vs {target.shape}")
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def network_inference(network, image_rgb, *, device: str = "cuda:0", max_pixels: int = 1024 * 1024):
    """直接用训练中的网络做一次超分（不走 RealESRGANer，便于验证）。

    Args:
        network: 已加载基础权重（并可含 LoRA）的网络。
        image_rgb: HxWx3 RGB uint8。
        device: 推理设备。
        max_pixels: 输入像素上限，超限时中心裁剪，防止验证阶段 OOM。

    Returns:
        超分后的 RGB uint8 数组。
    """
    if not HAS_TORCH:
        raise ValidatorError("验证需要安装 PyTorch。")
    if image_rgb is None or getattr(image_rgb, "size", 0) == 0:
        raise ValidatorError("验证输入为空。")

    image = image_rgb
    height, width = image.shape[:2]
    if height * width > max_pixels:
        scale = (max_pixels / float(height * width)) ** 0.5
        new_h, new_w = max(8, int(height * scale)), max(8, int(width * scale))
        top = max(0, (height - new_h) // 2)
        left = max(0, (width - new_w) // 2)
        image = image[top:top + new_h, left:left + new_w]

    was_training = network.training
    network.eval()
    tensor = (
        torch.from_numpy(np.ascontiguousarray(image).astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    param = next(network.parameters(), None)
    if param is not None and param.dtype == torch.float16 and tensor.dtype != torch.float16:
        tensor = tensor.half()
    try:
        with torch.no_grad():
            output = network(tensor)
    except Exception as exc:  # noqa: BLE001
        raise ValidatorError(f"验证推理失败：{exc}") from exc
    finally:
        if was_training:
            network.train()

    array = output.detach().float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return (array * 255.0).round().astype(np.uint8)


# --------------------------------------------------------------------------- #
# 拼图
# --------------------------------------------------------------------------- #
def _label(image, text: str, height: int = 22):
    """在图片上方加一条标签栏（英文，避免字体缺字）。"""
    if not HAS_CV2:
        return image
    bar = np.zeros((height, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (6, int(height * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, image])


def make_comparison(lr_rgb, sr_rgb, hr_rgb) -> np.ndarray:
    """生成 ``LR | SR | HR`` 三联对比图（高度对齐）。"""
    if not HAS_CV2 or not HAS_NUMPY:
        raise ValidatorError("生成对比图需要 opencv-python 与 numpy。")
    target_h, target_w = hr_rgb.shape[:2]

    def _fit(image):
        if image.shape[0] != target_h or image.shape[1] != target_w:
            return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return image

    panels = [
        _label(_fit(lr_rgb), "LR input"),
        _label(_fit(sr_rgb), "SR (LoRA)"),
        _label(_fit(hr_rgb), "HR target"),
    ]
    return np.hstack(panels)


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #
class Validator:
    """训练中定期验证：出预览图 + PSNR。"""

    def __init__(
        self,
        val_files: Sequence[Path],
        *,
        tag: str = "default",
        device: str = "cuda:0",
        scale: int | None = None,
        max_images: int = 4,
        crop_size: int = 160,
        seed: int = 1234,
    ):
        self.files = [Path(p) for p in val_files][: max(1, int(max_images))]
        self.tag = tag
        self.device = device
        self.scale = int(scale or 4)
        self.crop_size = int(crop_size)
        self.seed = int(seed)
        self.output_dir = validate_root(tag)
        self.history: list[dict] = []

    @property
    def available(self) -> bool:
        return bool(self.files) and HAS_TORCH and HAS_CV2

    def _prepare_pair(self, index: int):
        """准备一组固定的 (LR, HR) 验证对（跨 epoch 完全一致，保证可比）。"""
        path = self.files[index % len(self.files)]
        hr_full = ds.read_image_rgb(path)
        if hr_full is None:
            return None
        size = min(self.crop_size, hr_full.shape[0], hr_full.shape[1])
        rng = random.Random(self.seed + index)
        top = rng.randint(0, max(0, hr_full.shape[0] - size))
        left = rng.randint(0, max(0, hr_full.shape[1] - size))
        hr = np.ascontiguousarray(hr_full[top:top + size, left:left + size])
        cfg = ds.DegradationConfig(scale=self.scale)
        lr = ds.degrade_image(hr, cfg, random.Random(self.seed + index))
        return path.name, lr, hr

    def run(self, network, epoch: int) -> dict:
        """执行一次验证，返回 ``{"epoch", "psnr", "images": [...]}``。"""
        if not self.available:
            return {"epoch": epoch, "psnr": None, "images": [], "skipped": "验证数据或依赖不可用"}

        started = time.time()
        images: list[str] = []
        scores: list[float] = []
        for index in range(len(self.files)):
            prepared = self._prepare_pair(index)
            if prepared is None:
                continue
            name, lr, hr = prepared
            try:
                sr = network_inference(network, lr, device=self.device)
            except ValidatorError as exc:
                logger.warning("验证推理失败：%s", exc)
                continue
            if sr.shape[:2] != hr.shape[:2]:
                sr = cv2.resize(sr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)
            psnr = compute_psnr(sr, hr)
            scores.append(psnr)
            panel = make_comparison(lr, sr, hr)
            target = self.output_dir / f"epoch_{int(epoch):04d}_{index:02d}.png"
            try:
                cv2.imwrite(str(target), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                images.append(str(target))
            except Exception as exc:  # noqa: BLE001
                logger.warning("验证图保存失败：%s", exc)
            logger.info("验证 %s | epoch %d | PSNR %.2f dB", name, epoch, psnr)

        mean_psnr = round(float(sum(scores) / len(scores)), 3) if scores else None
        result = {
            "epoch": int(epoch),
            "psnr": mean_psnr,
            "images": images,
            "elapsed": round(time.time() - started, 2),
        }
        self.history.append(result)
        return result

    def best_psnr(self) -> float | None:
        """历史最佳 PSNR。"""
        values = [item["psnr"] for item in self.history if item.get("psnr") is not None]
        return max(values) if values else None

    def latest_images(self, limit: int = 6) -> list[str]:
        """最近若干张验证图路径（供 UI 画廊）。"""
        collected: list[str] = []
        for item in reversed(self.history):
            for path in reversed(item.get("images", [])):
                collected.append(path)
                if len(collected) >= limit:
                    return collected
        return collected

    def summary_text(self) -> str:
        """验证历史摘要（Markdown）。"""
        if not self.history:
            return "_暂无验证记录_"
        lines = ["| Epoch | PSNR (dB) |", "| --- | --- |"]
        for item in self.history[-20:]:
            psnr = item.get("psnr")
            lines.append(f"| {item['epoch']} | {'-' if psnr is None else f'{psnr:.2f}'} |")
        best = self.best_psnr()
        if best is not None:
            lines.append(f"\n最佳 PSNR：**{best:.2f} dB**")
        return "\n".join(lines)


def list_validation_images(tag: str = "default", limit: int = 8) -> list[str]:
    """列出某次训练已有的验证图（UI 刷新用）。"""
    directory = validate_root(tag)
    files = sorted(directory.glob("epoch_*.png"))
    return [str(p) for p in files[-limit:]]


__all__ = [
    "HAS_CV2",
    "HAS_TORCH",
    "ValidatorError",
    "validate_root",
    "compute_psnr",
    "network_inference",
    "make_comparison",
    "Validator",
    "list_validation_images",
]
