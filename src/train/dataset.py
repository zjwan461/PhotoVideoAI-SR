"""训练数据集：读取 HR 高清图，自动退化生成 LR 配对样本。

退化管线（可按需开关，默认模拟 Real-ESRGAN 的经典二阶退化）
----------------------------------------------------------
1. 随机缩放（bicubic / bilinear / area，scale 附近抖动）
2. 可选高斯模糊（随机核与 sigma）
3. 可选 sinc 振铃（模拟过强的下采样核）
4. 可选二次退化（再模糊 + 再下采样 + 加噪，顺序随机）
5. 可选 JPEG 压缩（随机质量）
6. 可选高斯噪声 / 灰度化

设计约定
--------
* 只读用户数据，**绝不修改原始 HR 文件**；
* 自动过滤「尺寸不足」与「解码失败」的图片，并记录原因供 UI 展示；
* 返回 ``float32`` 的 CHW 张量，值域 ``[0, 1]``；
* 未安装 torch 时模块仍可导入（仅数据增强函数可用）。
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.common import get_logger, list_image_files

logger = get_logger("dataset")

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
    from torch.utils.data import DataLoader, Dataset

    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    DataLoader = object  # type: ignore[assignment,misc]
    Dataset = object  # type: ignore[assignment,misc]
    HAS_TORCH = False


class DatasetError(RuntimeError):
    """数据集构建相关的可读错误。"""


def _require_cv2():
    if not HAS_CV2 or not HAS_NUMPY:
        raise DatasetError(
            "数据集需要 opencv-python 与 numpy，请执行：pip install -r requirements.txt"
        )
    return cv2


# --------------------------------------------------------------------------- #
# 退化配置与实现
# --------------------------------------------------------------------------- #
@dataclass
class DegradationConfig:
    """LR 退化管线参数（合理默认值即可直接训练）。"""

    scale: int = 4
    resize_modes: tuple[str, ...] = ("bicubic", "bilinear", "area")
    # 模糊
    blur_prob: float = 0.6
    blur_sigma: tuple[float, float] = (0.2, 2.0)
    blur_kernel: tuple[int, int] = (3, 9)
    # sinc 振铃
    sinc_prob: float = 0.3
    sinc_kernel: tuple[int, int] = (3, 9)
    # 二次退化
    second_degradation_prob: float = 0.5
    second_scale: tuple[float, float] = (0.7, 1.0)
    # JPEG 压缩
    jpeg_prob: float = 0.7
    jpeg_quality: tuple[int, int] = (35, 95)
    # 噪声与灰度
    noise_prob: float = 0.5
    noise_sigma: tuple[float, float] = (1.0, 12.0)
    gray_prob: float = 0.05
    # 每张 HR 图在一个 epoch 内被采样的次数
    repeat: int = 1

    def validate(self) -> None:
        if self.scale < 1:
            raise DatasetError("放大倍数 scale 必须 >= 1。")
        for name in ("blur_prob", "jpeg_prob", "noise_prob", "sinc_prob",
                     "second_degradation_prob", "gray_prob"):
            value = getattr(self, name)
            if not 0.0 <= float(value) <= 1.0:
                raise DatasetError(f"{name} 必须在 [0, 1] 范围内，当前 {value}。")


def _interp_flag(mode: str):
    cv2_module = _require_cv2()
    return {
        "bicubic": cv2_module.INTER_CUBIC,
        "bilinear": cv2_module.INTER_LINEAR,
        "area": cv2_module.INTER_AREA,
        "nearest": cv2_module.INTER_NEAREST,
        "lanczos": cv2_module.INTER_LANCZOS4,
    }.get(mode, cv2_module.INTER_CUBIC)


def _gaussian_blur(image, sigma: float, kernel: int):
    cv2_module = _require_cv2()
    size = int(kernel) | 1  # 保证奇数
    return cv2_module.GaussianBlur(image, (size, size), float(sigma))


def _sinc_kernel(size: int, omega: float):
    """构造 2D sinc 低通核，用于模拟过强下采样带来的振铃。"""
    size = int(size) | 1
    x = np.arange(-size // 2 + 1, size // 2 + 1, dtype=np.float32)
    kernel_1d = np.sinc(x * omega / np.pi) * omega / np.pi
    kernel = np.outer(kernel_1d, kernel_1d).astype(np.float32)
    total = kernel.sum()
    if total != 0:
        kernel = kernel / total
    return kernel


def _apply_sinc(image, kernel_size: int, omega: float):
    cv2_module = _require_cv2()
    kernel = _sinc_kernel(kernel_size, omega)
    return cv2_module.filter2D(
        image, -1, kernel, borderType=cv2_module.BORDER_REPLICATE
    )


def _add_noise(image, sigma: float, rng: random.Random):
    """加高斯噪声。

    标准库的 ``random.Random`` 没有向量化正态采样，这里从它派生一个 numpy 生成器
    （消耗一次 ``randrange``），从而在保持「给定种子可复现」的同时获得数组级采样。
    """
    seed = rng.randrange(1 << 32)
    nprng = np.random.default_rng(seed)
    noise = nprng.normal(0.0, float(sigma), image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _jpeg_compress(image, quality: int):
    cv2_module = _require_cv2()
    ok, buffer = cv2_module.imencode(".jpg", image, [int(cv2_module.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return image
    return cv2_module.imdecode(buffer, cv2_module.IMREAD_COLOR)


def degrade_image(hr_rgb, cfg: DegradationConfig | None = None, rng: random.Random | None = None):
    """把 HR RGB 图退化成 LR RGB 图（尺寸约为 ``1/scale``）。

    Args:
        hr_rgb: HxWx3 的 RGB uint8 数组。
        cfg: 退化配置，None 使用默认值。
        rng: 随机源（便于复现）。

    Returns:
        LR 的 RGB uint8 数组。
    """
    _require_cv2()
    cfg = cfg or DegradationConfig()
    rng = rng or random.Random()
    if hr_rgb is None or getattr(hr_rgb, "size", 0) == 0:
        raise DatasetError("退化输入为空。")

    image = hr_rgb
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    image = np.ascontiguousarray(image)

    height, width = image.shape[:2]

    # 1) 可选模糊
    blurred = image
    if rng.random() < cfg.blur_prob:
        blurred = _gaussian_blur(
            image,
            rng.uniform(*cfg.blur_sigma),
            rng.randint(*cfg.blur_kernel),
        )

    # 2) 可选 sinc 振铃
    if rng.random() < cfg.sinc_prob:
        blurred = _apply_sinc(blurred, rng.randint(*cfg.sinc_kernel), rng.uniform(0.5, 1.0))

    # 3) 下采样（在 scale 附近抖动，模拟真实倍率不确定）
    mode = rng.choice(cfg.resize_modes)
    out_w = max(1, int(round(width / cfg.scale)))
    out_h = max(1, int(round(height / cfg.scale)))
    lr = cv2.resize(blurred, (out_w, out_h), interpolation=_interp_flag(mode))

    # 4) 二次退化：再模糊 + 再下采样 + 加噪，顺序随机
    if rng.random() < cfg.second_degradation_prob:
        steps = ["blur", "resize", "noise"]
        rng.shuffle(steps)
        for step in steps:
            if step == "blur":
                lr = _gaussian_blur(lr, rng.uniform(0.2, 1.5), rng.choice([3, 5]))
            elif step == "resize":
                factor = rng.uniform(*cfg.second_scale)
                new_size = (max(1, int(round(out_w * factor))), max(1, int(round(out_h * factor))))
                lr = cv2.resize(lr, new_size, interpolation=_interp_flag(rng.choice(cfg.resize_modes)))
                lr = cv2.resize(lr, (out_w, out_h), interpolation=_interp_flag("bicubic"))
            else:
                lr = _add_noise(lr, rng.uniform(1.0, 8.0), rng)

    # 5) JPEG 压缩
    if rng.random() < cfg.jpeg_prob:
        lr = _jpeg_compress(lr, rng.randint(*cfg.jpeg_quality))

    # 6) 噪声 / 灰度
    if rng.random() < cfg.noise_prob:
        lr = _add_noise(lr, rng.uniform(*cfg.noise_sigma), rng)
    if rng.random() < cfg.gray_prob:
        gray = cv2.cvtColor(lr, cv2.COLOR_RGB2GRAY)
        lr = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    return np.ascontiguousarray(lr)


# --------------------------------------------------------------------------- #
# 读写与校验
# --------------------------------------------------------------------------- #
def read_image_rgb(path: str | Path):
    """读取图片为 RGB；失败返回 None（用于过滤损坏文件）。"""
    _require_cv2()
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    except Exception as exc:  # noqa: BLE001
        logger.debug("解码失败：%s -> %s", path, exc)
        return None
    if image is None or image.size == 0:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _resize_to(image, side: int):
    """把方图缩放到指定边长（缩小用 AREA 抗锯齿，放大用 CUBIC）。"""
    _require_cv2()
    if image.shape[0] == side and image.shape[1] == side:
        return image
    shrinking = side < min(image.shape[:2])
    flag = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return np.ascontiguousarray(cv2.resize(image, (side, side), interpolation=flag))


def augment_pair(hr, lr, rng: random.Random):
    """同步几何增强：随机水平/垂直翻转 + 90 度旋转。"""
    _require_cv2()
    if rng.random() < 0.5:
        hr = np.ascontiguousarray(hr[:, ::-1])
        lr = np.ascontiguousarray(lr[:, ::-1])
    if rng.random() < 0.5:
        hr = np.ascontiguousarray(hr[::-1, :])
        lr = np.ascontiguousarray(lr[::-1, :])
    rotations = rng.randint(0, 3)
    if rotations:
        hr = np.ascontiguousarray(np.rot90(hr, rotations))
        lr = np.ascontiguousarray(np.rot90(lr, rotations))
    return hr, lr


def to_tensor(image_rgb):
    """HWC uint8 RGB -> CHW float32 [0,1]。"""
    if not HAS_TORCH:
        raise DatasetError("转换为张量需要安装 PyTorch。")
    array = np.ascontiguousarray(image_rgb).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def collect_hr_files(inputs: str | Path | Sequence[str | Path]) -> list[Path]:
    """收集 HR 图片：支持目录、单文件、文件列表混合输入。"""
    files: list[Path] = []
    items = [inputs] if isinstance(inputs, (str, Path)) else list(inputs)
    for item in items:
        path = Path(item)
        if path.is_dir():
            files.extend(list_image_files(path))
        elif path.is_file():
            files.append(path)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in files:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def split_train_val(
    files: Sequence[Path], val_ratio: float = 0.05, *, seed: int = 0, min_val: int = 1
) -> tuple[list[Path], list[Path]]:
    """按比例切分训练/验证集（验证集至少 ``min_val`` 张，前提是总量够）。"""
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    if not shuffled:
        return [], []
    val_count = int(round(len(shuffled) * max(0.0, min(0.5, val_ratio))))
    if len(shuffled) > 1:
        val_count = max(min_val, val_count)
    val_count = min(val_count, max(0, len(shuffled) - 1)) if len(shuffled) > 1 else 0
    val = shuffled[:val_count]
    train = shuffled[val_count:] or shuffled
    return train, val


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
if HAS_TORCH:

    class PairedSRDataset(Dataset):  # type: ignore[misc]
        """HR -> (LR, HR) 配对数据集。

        每张 HR 图随机裁剪 ``patch_size`` 大小，再退化生成 LR；
        ``repeat`` 控制同一张图在一个 epoch 内被采样多少次。
        """

        def __init__(
            self,
            inputs: str | Path | Sequence[str | Path] | None = None,
            *,
            patch_size: int = 128,
            cfg: DegradationConfig | None = None,
            augment: bool = True,
            seed: int = 0,
            files: Sequence[Path] | None = None,
            min_hr_size: int | None = None,
        ):
            self.cfg = cfg or DegradationConfig()
            self.cfg.validate()
            self.patch_size = int(patch_size)
            self.augment = bool(augment)
            self.seed = int(seed)
            self.epoch = 0
            self.skipped: list[str] = []

            candidates = list(files) if files is not None else collect_hr_files(inputs)  # type: ignore[arg-type]
            if not candidates:
                raise DatasetError("没有找到任何 HR 图片，请检查数据集路径。")

            min_size = int(min_hr_size if min_hr_size is not None else patch_size + 8)
            self.files: list[Path] = []
            for path in candidates:
                image = read_image_rgb(path)
                if image is None:
                    self.skipped.append(f"{path.name}: 文件损坏或格式不支持")
                    continue
                if min(image.shape[0], image.shape[1]) < min_size:
                    self.skipped.append(
                        f"{path.name}: 分辨率过小 {image.shape[1]}x{image.shape[0]}（需 >= {min_size}）"
                    )
                    continue
                self.files.append(path)

            if not self.files:
                raise DatasetError(
                    "所有候选图片都被过滤掉了。\n" + "\n".join(self.skipped[:10])
                )
            if self.skipped:
                logger.warning("已过滤 %d 张不可用图片", len(self.skipped))

        # -- 采样 ---------------------------------------------------------- #
        def set_epoch(self, epoch: int) -> None:
            """每个 epoch 更新随机种子，保证重复采样时样本不固定。"""
            self.epoch = int(epoch)

        def __len__(self) -> int:
            return len(self.files) * max(1, int(self.cfg.repeat))

        def _rng(self, index: int) -> random.Random:
            return random.Random(self.seed + self.epoch * 100003 + index)

        def __getitem__(self, index: int):
            file_index = int(index) % len(self.files)
            rng = self._rng(int(index))
            hr_full = read_image_rgb(self.files[file_index])
            if hr_full is None:  # 运行期文件被删/损坏时兜底重试一次
                hr_full = read_image_rgb(self.files[(file_index + 1) % len(self.files)])
            if hr_full is None:
                raise DatasetError("数据集中的图片无法解码，请检查数据完整性。")

            # 轻微尺寸抖动 + 回缩（Real-ESRGAN 的 resize-back 技巧）：
            # 裁剪略大的区域再缩回标称尺寸，可带来温和的倍率扰动，同时保证 batch 形状一致。
            height, width = hr_full.shape[:2]
            crop = min(self.patch_size + rng.randint(0, 2) * 8, height, width)
            top = rng.randint(0, max(0, height - crop))
            left = rng.randint(0, max(0, width - crop))
            hr_patch = np.ascontiguousarray(hr_full[top:top + crop, left:left + crop])

            lr_patch = degrade_image(hr_patch, self.cfg, rng)
            hr_patch = _resize_to(hr_patch, self.patch_size)
            lr_side = max(1, self.patch_size // max(1, int(self.cfg.scale)))
            lr_patch = _resize_to(lr_patch, lr_side)

            if self.augment:
                hr_patch, lr_patch = augment_pair(hr_patch, lr_patch, rng)

            return to_tensor(lr_patch), to_tensor(hr_patch)


def build_dataset(
    inputs: str | Path | Sequence[str | Path],
    *,
    patch_size: int = 128,
    cfg: DegradationConfig | None = None,
    val_ratio: float = 0.05,
    augment: bool = True,
    seed: int = 0,
) -> tuple[PairedSRDataset, PairedSRDataset | None, dict]:
    """构建训练（与可选验证）数据集，返回 ``(train, val, summary)``。"""
    if not HAS_TORCH:
        raise DatasetError("构建数据集需要安装 PyTorch。")
    files = collect_hr_files(inputs)
    if not files:
        raise DatasetError(f"目录中没有图片：{inputs}")
    train_files, val_files = split_train_val(files, val_ratio, seed=seed)

    train_set = PairedSRDataset(
        files=train_files, patch_size=patch_size, cfg=cfg, augment=augment, seed=seed
    )
    val_set = None
    if val_files:
        try:
            val_set = PairedSRDataset(
                files=val_files, patch_size=patch_size, cfg=cfg, augment=False, seed=seed + 1
            )
        except DatasetError as exc:  # 验证集全被过滤时不影响训练
            logger.warning("验证集构建失败，已跳过验证：%s", exc)

    summary = {
        "train_files": len(train_set.files),
        "val_files": len(val_set.files) if val_set else 0,
        "skipped": train_set.skipped + (val_set.skipped if val_set else []),
        "patch_size": patch_size,
        "scale": (cfg or DegradationConfig()).scale,
        "train_samples": len(train_set),
        "val_samples": len(val_set) if val_set else 0,
    }
    logger.info(
        "数据集就绪：训练 %d 张 / 验证 %d 张（跳过 %d 张，patch=%d，scale=%d）",
        summary["train_files"], summary["val_files"], len(summary["skipped"]),
        patch_size, summary["scale"],
    )
    return train_set, val_set, summary


def build_dataloader(
    dataset,
    *,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    persistent_workers: bool | None = None,
):
    """构建 DataLoader（Windows 下建议 num_workers=0 以避免多进程问题）。"""
    if not HAS_TORCH:
        raise DatasetError("构建 DataLoader 需要安装 PyTorch。")
    workers = max(0, int(num_workers))
    if persistent_workers is None:
        persistent_workers = workers > 0
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=bool(shuffle),
        num_workers=workers,
        pin_memory=bool(pin_memory) and HAS_TORCH,
        drop_last=bool(drop_last),
        persistent_workers=bool(persistent_workers) and workers > 0,
    )


__all__ = [
    "HAS_CV2",
    "HAS_TORCH",
    "DatasetError",
    "DegradationConfig",
    "degrade_image",
    "read_image_rgb",
    "augment_pair",
    "_resize_to",
    "to_tensor",
    "collect_hr_files",
    "split_train_val",
    "PairedSRDataset",
    "build_dataset",
    "build_dataloader",
]
