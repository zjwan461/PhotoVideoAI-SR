"""LoRA 微调训练循环：loss、优化器、进度回调、断点续训、验证与保存。

模块结构
--------
* :class:`TrainConfig` —— 全部超参（dataclass，便于 UI 组装与日志留档）
* :class:`Trainer` —— 同步训练循环，通过回调向外汇报进度
* :class:`TrainingRunner` —— 把 :class:`Trainer` 放进后台线程运行，
  暴露 :meth:`TrainingRunner.status` / :meth:`TrainingRunner.stop`，
  供 Gradio 一边轮询状态、一边点“停止”而不阻塞队列

关键行为
--------
* 只训练 LoRA 参数，基础权重冻结；导出文件只含 LoRA（永不改写 ``model_weights/``）；
* 支持 AMP、梯度累积、梯度裁剪；
* 每 ``val_interval`` 个 epoch 做验证（预览图 + PSNR），每 ``save_interval`` 存一次 checkpoint；
* 断点续训：``resume_from`` 指向 ``*_state.pt`` 即可恢复 epoch / 优化器 / 最佳 PSNR；
* 显存不足会转成可读建议，而不是把 CUDA 堆栈抛给用户；
* 通过 ``should_stop`` 回调（或 :meth:`TrainingRunner.stop`）可随时优雅停止并保存当前进度。
"""

from __future__ import annotations

import json
import math
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import config
from src.common import (
    clear_cuda_cache,
    elapsed_str,
    friendly_cuda_error,
    get_free_vram_gb,
    get_logger,
)
from src.train import dataset as ds
from src.train import lora_network as ln
from src.train import lora_utils
from src.train import validator as val_mod

logger = get_logger("trainer")

try:
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

try:  # 感知损失为可选项
    import lpips

    HAS_LPIPS = True
except Exception:  # pragma: no cover
    lpips = None  # type: ignore[assignment]
    HAS_LPIPS = False


class TrainingError(RuntimeError):
    """训练相关的可读错误。"""


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """一次 LoRA 微调的全部参数。"""

    # 数据与命名
    dataset_paths: list[str] = field(default_factory=list)
    lora_name: str = "my_lora"
    tag: str = ""                       # 日志/验证图分组名，默认用 lora_name
    # 训练超参
    epochs: int = 10
    batch_size: int = 4
    patch_size: int = 128
    learning_rate: float = 1e-4
    rank: int = 8
    alpha: float | None = None
    weight_decay: float = 0.0
    grad_accum: int = 1
    grad_clip: float = 1.0
    amp: bool = True
    # 模型
    base_model: str = config.DEFAULT_MODEL_NAME
    auto_download: bool = True
    scale: int = 4
    # 验证与保存
    val_ratio: float = 0.05
    val_interval: int = 1
    val_images: int = 4
    save_interval: int = 5
    keep_last_checkpoints: int = 3
    # 损失
    loss: str = "l1"                    # l1 | l1+perceptual
    perceptual_weight: float = 0.1
    # 运行
    resume_from: str | None = None
    device: str = "cuda:0"
    num_workers: int = 0
    seed: int = 0
    log_every: int = 5                  # 每多少个 step 汇报一次进度

    def validate(self) -> None:
        if not self.dataset_paths:
            raise TrainingError("请先指定训练数据集（高清图片文件夹或文件列表）。")
        if self.epochs < 1:
            raise TrainingError("epoch 必须 >= 1。")
        if self.batch_size < 1:
            raise TrainingError("batch size 必须 >= 1。")
        if self.patch_size < 32:
            raise TrainingError("裁剪尺寸建议 >= 32（过小会让退化/放大失去意义）。")
        if self.learning_rate <= 0:
            raise TrainingError("学习率必须 > 0。")
        if not 1 <= self.rank <= 256:
            raise TrainingError("LoRA rank 建议在 1~256 之间。")
        if self.grad_accum < 1:
            raise TrainingError("梯度累积步数必须 >= 1。")
        if self.loss not in ("l1", "l1+perceptual"):
            raise TrainingError(f"未知的损失类型：{self.loss}")
        # device 可能来自 UI 下拉标签（"cuda:0 | <名称> (32 GB, sm_89)"）：
        # 在这里统一成 torch 可接受的写法，后续 .to(cfg.device) 才不会报错。
        self.device = config.normalize_device(self.device)

    @property
    def run_tag(self) -> str:
        return self.tag or self.lora_name

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_training_vram_gb(
    patch_size: int = 128, batch_size: int = 4, rank: int = 8, *, num_block: int = 23
) -> float:
    """粗略估算 LoRA 训练显存需求（GiB），用于训练前的友好提示。"""
    patch_mb = (int(patch_size) ** 2) * int(batch_size) * 3 * 4 / (1024 ** 2)  # 单份激活
    model_gb = 2.2 + 0.055 * int(num_block)
    activation_gb = patch_mb / 1024.0 * (12.0 + 0.002 * int(num_block))
    return round(model_gb + activation_gb + 0.05 * int(rank), 1)


# --------------------------------------------------------------------------- #
# 训练器
# --------------------------------------------------------------------------- #
class Trainer:
    """同步训练循环；所有对外通信通过回调完成。"""

    def __init__(
        self,
        cfg: TrainConfig,
        *,
        log_cb: Callable[[str], None] | None = None,
        status_cb: Callable[[dict], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        network_factory=None,
    ):
        if not HAS_TORCH:
            raise TrainingError("LoRA 微调需要安装 PyTorch（CUDA 版）。")
        cfg.validate()
        self.cfg = cfg
        self.log_cb = log_cb or (lambda message: logger.info("%s", message))
        self.status_cb = status_cb or (lambda status: None)
        self.should_stop = should_stop or (lambda: False)
        self.network_factory = network_factory

        self.model = None
        self.optimizer = None
        self.scaler = None
        self.criterion = None
        self.validator: val_mod.Validator | None = None
        self.output_dir = config.LOG_DIR / "train" / self._safe(self.cfg.run_tag)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.final_path: Path | None = None
        self.state_path: Path | None = None
        self.start_epoch = 0
        self.best_psnr: float | None = None
        self.step = 0
        self.total_steps = 0
        self.history: list[dict] = []
        self.started_at = 0.0

    # -- 工具 ------------------------------------------------------------ #
    @staticmethod
    def _safe(text: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text)) or "run"

    def log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        logger.info("%s", message)
        self.log_cb(line)

    def _emit_status(self, **extra) -> None:
        status = self.status_snapshot()
        status.update(extra)
        self.status_cb(status)

    def status_snapshot(self) -> dict:
        """当前训练状态（供 UI 轮询）。"""
        elapsed = time.time() - self.started_at if self.started_at else 0.0
        done = self.step
        remaining_ratio = 1.0 - (done / self.total_steps) if self.total_steps else 0.0
        eta = (elapsed / done * (self.total_steps - done)) if done and self.total_steps else None
        return {
            "epoch": self._current_epoch,
            "total_epochs": self.cfg.epochs,
            "step": done,
            "total_steps": self.total_steps,
            "loss": self._last_loss,
            "avg_loss": self._avg_loss,
            "lr": self._current_lr,
            "psnr": self.best_psnr,
            "elapsed": elapsed,
            "elapsed_text": elapsed_str(elapsed),
            "eta_text": elapsed_str(eta) if eta else "-",
            "remaining_ratio": round(remaining_ratio, 4),
            "free_vram_gb": get_free_vram_gb(config.gpu_index_from_option(self.cfg.device)),
            "output": str(self.final_path) if self.final_path else None,
            "device": self.cfg.device,
        }

    _current_epoch = 0
    _last_loss: float | None = None
    _avg_loss: float | None = None
    _current_lr: float | None = None
    _loss_sum = 0.0
    _loss_count = 0

    # -- 构建 ------------------------------------------------------------ #
    def _build_loss(self):
        cfg = self.cfg
        if cfg.loss == "l1+perceptual":
            if not HAS_LPIPS:
                self.log("⚠️ 未安装 lpips，感知损失不可用，已自动回退为纯 L1（pip install lpips 可启用）")
                return torch.nn.L1Loss(), None, 0.0
            try:
                perceptual = lpips.LPIPS(net="vgg", verbose=False).to(cfg.device)
                perceptual.eval()
                for parameter in perceptual.parameters():
                    parameter.requires_grad_(False)
                self.log(f"感知损失已启用：lpips(vgg)，权重 {cfg.perceptual_weight:.3f}")
                return torch.nn.L1Loss(), perceptual, float(cfg.perceptual_weight)
            except Exception as exc:  # noqa: BLE001
                self.log(f"⚠️ 感知损失初始化失败（{exc}），回退为纯 L1")
                return torch.nn.L1Loss(), None, 0.0
        return torch.nn.L1Loss(), None, 0.0

    def _build_optimizer(self):
        cfg = self.cfg
        try:
            groups = ln.parameter_groups(self.model, weight_decay=cfg.weight_decay)
        except RuntimeError as exc:
            raise TrainingError(str(exc)) from exc
        return torch.optim.AdamW(groups, lr=float(cfg.learning_rate), betas=(0.9, 0.99))

    def _build_scaler(self):
        cfg = self.cfg
        enabled = bool(cfg.amp) and str(cfg.device).startswith("cuda")
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)  # torch >= 2.4
        except (AttributeError, TypeError):
            try:
                return torch.cuda.amp.GradScaler(enabled=enabled)
            except Exception:  # noqa: BLE001
                return None

    def _autocast(self):
        cfg = self.cfg
        enabled = bool(cfg.amp) and str(cfg.device).startswith("cuda")
        try:
            return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
        except Exception:  # noqa: BLE001
            from contextlib import nullcontext

            return nullcontext()

    # -- 断点续训 -------------------------------------------------------- #
    def maybe_resume(self) -> None:
        """恢复优化器状态、epoch 与最佳 PSNR。"""
        resume = self.cfg.resume_from
        if not resume:
            return
        path = Path(resume)
        if not path.is_file():
            self.log(f"⚠️ 找不到续训检查点 {path}，将从头开始训练")
            return
        try:
            payload = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            self.log(f"⚠️ 读取检查点失败（{exc}），将从头开始训练")
            return
        if not isinstance(payload, dict) or "lora_state_dict" not in payload:
            self.log(f"⚠️ {path.name} 不是续训检查点（缺少 lora_state_dict），已忽略")
            return
        try:
            lora_utils.lora_runtime.load_lora_weights(self.model, path)
        except Exception as exc:  # noqa: BLE001
            self.log(f"⚠️ 加载 LoRA 权重失败（{exc}），将从头开始训练")
            return

        state = payload.get("trainer_state") or {}
        if state.get("optimizer") and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except Exception as exc:  # noqa: BLE001
                self.log(f"⚠️ 优化器状态恢复失败（{exc}），继续训练但动量会重置")
        self.start_epoch = int(state.get("epoch", 0))
        self.best_psnr = state.get("best_psnr")
        self.log(f"已恢复训练状态：从 epoch {self.start_epoch + 1} 继续，最佳 PSNR {self.best_psnr}")

    def _save_checkpoint(self, epoch: int, *, is_final: bool = False, is_best: bool = False) -> Path:
        """保存 LoRA 检查点（含可续训状态，便于 resume）。"""
        assert self.final_path is not None
        if is_final:
            target = self.final_path
        elif is_best:
            target = self.final_path.with_name(f"{self.final_path.stem}_best.pth")
        else:
            target = self.final_path.with_name(f"{self.final_path.stem}_epoch{epoch:04d}.pth")

        lora_utils.save_lora(
            self.model,
            target.stem,
            rank=self.cfg.rank,
            alpha=self.cfg.alpha,
            base_model=self.cfg.base_model,
            meta={
                "epoch": int(epoch),
                "tag": self.cfg.run_tag,
                "config": self.cfg.to_dict(),
                "best_psnr": self.best_psnr,
            },
            ensure_unique=False,
        )

        state_payload = {
            "format": "photovideoai-sr-lora-state",
            "epoch": int(epoch),
            "best_psnr": self.best_psnr,
            "lora_state_dict": lora_utils.lora_runtime.lora_state_dict(self.model),
            "trainer_state": {
                "optimizer": self.optimizer.state_dict() if self.optimizer else None,
                "scaler": self.scaler.state_dict() if self.scaler is not None else None,
                "epoch": int(epoch),
                "best_psnr": self.best_psnr,
            },
        }
        # 续训状态放到 logs/train/<tag>/，让 lora_weights/ 只保留可部署的 LoRA 权重
        self.output_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.output_dir / f"{target.stem}.state.pt"
        torch.save(state_payload, str(state_path))
        self.state_path = state_path
        size_mb = target.stat().st_size / (1024 ** 2)
        self.log(f"💾 已保存检查点 {target.name}（{size_mb:.2f} MB）+ 续训状态 {state_path.name}")
        self._prune_checkpoints(keep=target.name)
        return target

    def _prune_checkpoints(self, *, keep: str) -> None:
        """只保留最近若干个 epoch 检查点，避免磁盘无限增长。"""
        assert self.final_path is not None
        limit = max(1, int(self.cfg.keep_last_checkpoints))
        pattern = f"{self.final_path.stem}_epoch*.pth"
        checkpoints = sorted(
            self.final_path.parent.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True
        )
        for path in checkpoints[limit:]:
            try:
                path.unlink()
                (self.output_dir / f"{path.stem}.state.pt").unlink(missing_ok=True)
                self.log(f"🧹 清理旧检查点 {path.name}")
            except OSError:
                continue

    # -- 主循环 ---------------------------------------------------------- #
    def run(self) -> dict:
        """执行训练，返回结果摘要字典。"""
        cfg = self.cfg
        self.started_at = time.time()

        torch.manual_seed(cfg.seed)
        try:
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.seed)
        except Exception:  # noqa: BLE001
            pass

        if not str(cfg.device).startswith("cuda"):
            self.log("⚠️ 当前不是 CUDA 设备：仅建议用于冒烟测试，正式微调请使用 GPU。")
            cfg.amp = False

        # ---- 数据 ----
        self.log(f"📂 加载数据集：{'、'.join(cfg.dataset_paths)}")
        start = time.time()
        train_set, val_set, summary = ds.build_dataset(
            cfg.dataset_paths,
            patch_size=cfg.patch_size,
            cfg=ds.DegradationConfig(scale=cfg.scale),
            val_ratio=cfg.val_ratio,
            augment=True,
            seed=cfg.seed,
        )
        self.log(
            f"✅ 数据集就绪：训练 {summary['train_files']} 张 / 验证 {summary['val_files']} 张"
            f"（跳过 {len(summary['skipped'])} 张，用时 {time.time() - start:.1f}s）"
        )
        for reason in summary["skipped"][:5]:
            self.log(f"   · 已跳过 {reason}")

        loader = ds.build_dataloader(
            train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
        )

        # ---- 模型 ----
        self.log(f"🧠 构建模型：{cfg.base_model} + LoRA(rank={cfg.rank}, alpha={cfg.alpha or cfg.rank})")
        self.model, weights, info = lora_utils.prepare_lora_model(
            cfg.base_model,
            rank=cfg.rank,
            alpha=cfg.alpha,
            device=cfg.device,
            auto_download=cfg.auto_download,
            network_factory=self.network_factory,
        )
        self.log(
            f"✅ LoRA 注入 {info['layers']} 层，可训练参数 {info['lora_params']:,}"
            f" / 总计 {info['total_params']:,}（{info['ratio_percent']:.3f}%）"
        )
        if weights:
            self.log(f"   基础权重（只读）：{weights}")

        self.final_path = lora_utils.lora_output_path(cfg.lora_name, ensure_unique=True)
        self.log(f"🎯 本次导出目标：{self.final_path.name}")

        self.optimizer = self._build_optimizer()
        self.scaler = self._build_scaler()
        self.criterion, perceptual, perceptual_weight = self._build_loss()
        self.validator = val_mod.Validator(
            [p for p in (val_set.files if val_set else [])],
            tag=cfg.run_tag,
            device=cfg.device,
            scale=cfg.scale,
            max_images=cfg.val_images,
        )
        self.maybe_resume()

        steps_per_epoch = max(1, math.ceil(len(loader) / max(1, cfg.grad_accum)))
        self.total_steps = steps_per_epoch * cfg.epochs
        self.log(f"🚀 开始训练：{cfg.epochs} epoch × {len(loader)} batch（共约 {self.total_steps} step）")

        stopped_early = False
        for epoch in range(self.start_epoch, cfg.epochs):
            self._current_epoch = epoch + 1
            train_set.set_epoch(epoch)
            epoch_loss, epoch_steps, stopped_early = self._run_epoch(
                loader, epoch, perceptual, perceptual_weight
            )
            self._avg_loss = round(epoch_loss / max(1, epoch_steps), 6)
            self.log(
                f"📈 epoch {epoch + 1}/{cfg.epochs} 完成：平均 loss {self._avg_loss:.6f}"
                f"（{epoch_steps} 个优化步）"
            )
            self.history.append({"epoch": epoch + 1, "loss": self._avg_loss})

            # ---- 验证 ----
            if self.validator and self.validator.files and (epoch + 1) % max(1, cfg.val_interval) == 0:
                result = self.validator.run(self.model, epoch + 1)
                if result.get("psnr") is not None:
                    self.log(f"🔍 验证 PSNR {result['psnr']:.2f} dB（{len(result['images'])} 张预览图）")
                    if self.best_psnr is None or result["psnr"] > self.best_psnr:
                        self.best_psnr = result["psnr"]
                        self._save_checkpoint(epoch + 1, is_best=True)

            # ---- 定期保存 ----
            if (epoch + 1) % max(1, cfg.save_interval) == 0:
                self._save_checkpoint(epoch + 1)

            self._emit_status()
            clear_cuda_cache()
            if stopped_early or self.should_stop():
                self.log("⏹ 收到停止请求，正在保存当前进度…")
                break

        final_path = self._save_checkpoint(self._current_epoch or cfg.epochs, is_final=True)
        self._write_summary(summary, final_path)
        elapsed = time.time() - self.started_at
        self.log(
            f"🎉 训练结束：用时 {elapsed_str(elapsed)}，最终 LoRA → {final_path}"
            + (f"，最佳 PSNR {self.best_psnr:.2f} dB" if self.best_psnr else "")
        )
        self._emit_status(done=True, stopped=stopped_early)

        return {
            "output_path": str(final_path),
            "state_path": str(self.state_path) if self.state_path else None,
            "best_psnr": self.best_psnr,
            "elapsed": elapsed,
            "epochs": self._current_epoch,
            "stopped_early": stopped_early,
            "dataset": summary,
            "history": self.history,
            "final_size_mb": round(final_path.stat().st_size / (1024 ** 2), 2),
        }

    def _run_epoch(self, loader, epoch: int, perceptual, perceptual_weight) -> tuple[float, int, bool]:
        """跑一个 epoch，返回 ``(loss 总和, 优化步数, 是否被停止)``。"""
        cfg = self.cfg
        assert self.model is not None and self.optimizer is not None
        self.model.train()  # 冻结的基础权重不会更新，LoRA 分支需要 train 行为

        loss_sum = 0.0
        step_count = 0
        accum = max(1, cfg.grad_accum)
        self.optimizer.zero_grad(set_to_none=True)
        epoch_start = time.time()

        for batch_index, (lr_batch, hr_batch) in enumerate(loader, start=1):
            if self.should_stop():
                return loss_sum, step_count, True
            lr_batch = lr_batch.to(cfg.device, non_blocking=True)
            hr_batch = hr_batch.to(cfg.device, non_blocking=True)

            try:
                with self._autocast():
                    sr = self.model(lr_batch)
                    loss = self.criterion(sr, hr_batch)
                    if perceptual is not None and perceptual_weight > 0:
                        loss = loss + perceptual_weight * perceptual(sr, hr_batch).mean()
                if self.scaler is not None and self.scaler.is_enabled():
                    self.scaler.scale(loss / accum).backward()
                else:
                    (loss / accum).backward()
            except Exception as exc:  # noqa: BLE001
                raise self._wrap_oom(exc) from exc

            if batch_index % accum == 0:
                try:
                    if self.scaler is not None and self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)
                        if cfg.grad_clip:
                            torch.nn.utils.clip_grad_norm_(ln.lora_parameters(self.model), cfg.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        if cfg.grad_clip:
                            torch.nn.utils.clip_grad_norm_(ln.lora_parameters(self.model), cfg.grad_clip)
                        self.optimizer.step()
                except Exception as exc:  # noqa: BLE001
                    raise self._wrap_oom(exc) from exc
                self.optimizer.zero_grad(set_to_none=True)
                step_count += 1
                self.step += 1
                self._current_lr = self.optimizer.param_groups[0]["lr"]
                self._loss_sum += float(loss.detach())
                self._loss_count += 1

            self._last_loss = round(float(loss.detach()), 6)
            loss_sum += float(loss.detach())

            if step_count and step_count % max(1, cfg.log_every) == 0:
                speed = (time.time() - epoch_start) / max(1, step_count)
                self._emit_status(
                    message=(
                        f"epoch {epoch + 1}/{cfg.epochs} | step {self.step}/{self.total_steps} | "
                        f"loss {self._last_loss:.6f} | {speed:.2f}s/step"
                    )
                )

        return loss_sum, max(1, step_count), False

    def _wrap_oom(self, exc: BaseException) -> BaseException:
        """把 CUDA OOM 转成带建议的可读异常。"""
        friendly = friendly_cuda_error(exc)
        if friendly:
            return TrainingError(
                friendly
                + "\n训练场景可尝试：调小「裁剪尺寸 patch」或「batch size」、"
                "关闭感知损失、降低 LoRA rank、开启梯度累积以减小显存峰值。"
            )
        return TrainingError(f"训练中断：{exc}")

    def _write_summary(self, dataset_summary: dict, final_path: Path) -> None:
        """把本次训练的参数与曲线写入 logs/train/<tag>/summary.json。"""
        payload = {
            "config": self.cfg.to_dict(),
            "dataset": dataset_summary,
            "history": self.history,
            "best_psnr": self.best_psnr,
            "output": str(final_path),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        target = self.output_dir / "summary.json"
        try:
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self.log(f"🧾 训练摘要已写入 {target}")
        except OSError as exc:
            self.log(f"⚠️ 训练摘要写入失败：{exc}")


# --------------------------------------------------------------------------- #
# 后台运行器（供 Gradio 使用）
# --------------------------------------------------------------------------- #
class TrainingRunner:
    """在后台线程中运行 :class:`Trainer`，对外暴露状态查询与停止开关。"""

    def __init__(self, cfg: TrainConfig, *, network_factory=None):
        self.cfg = cfg
        self.network_factory = network_factory
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._logs: list[str] = []
        self._status: dict = {"message": "尚未开始", "step": 0, "total_steps": 0}
        self._result: dict | None = None
        self._error: str | None = None
        self._trainer: Trainer | None = None

    # -- 回调 ------------------------------------------------------------ #
    def _on_log(self, line: str) -> None:
        with self._lock:
            self._logs.append(line)
            if len(self._logs) > 500:
                del self._logs[:100]

    def _on_status(self, status: dict) -> None:
        with self._lock:
            merged = dict(status)
            merged.setdefault("message", self._status.get("message", ""))
            self._status = merged

    def _should_stop(self) -> bool:
        return self._stop_event.is_set()

    # -- 控制 ------------------------------------------------------------ #
    def start(self) -> None:
        """启动后台训练线程（重复调用无效）。"""
        if self._thread is not None and self._thread.is_alive():
            raise TrainingError("已有训练任务在运行中，请先停止或等待其结束。")
        self._stop_event.clear()
        self._error = None
        self._result = None
        self._thread = threading.Thread(target=self._run, name="lora-trainer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._trainer = Trainer(
                self.cfg,
                log_cb=self._on_log,
                status_cb=self._on_status,
                should_stop=self._should_stop,
                network_factory=self.network_factory,
            )
            self._result = self._trainer.run()
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            logger.error("训练失败：%s\n%s", exc, traceback.format_exc())
            self._on_log(f"❌ 训练失败：{exc}")
        finally:
            with self._lock:
                self._status = dict(self._status)
                self._status["running"] = False

    def stop(self) -> None:
        """请求停止（训练循环会在下一个 step 边界退出并保存进度）。"""
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # -- 查询 ------------------------------------------------------------ #
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def result(self) -> dict | None:
        return self._result

    def logs(self) -> list[str]:
        with self._lock:
            return list(self._logs)

    def status(self) -> dict:
        with self._lock:
            status = dict(self._status)
        status["running"] = self.running
        status["error"] = self._error
        status["finished"] = (not self.running) and (self._result is not None or self._error is not None)
        return status

    def render_log(self, limit: int = 200) -> str:
        """把日志渲染成等宽文本块，供 UI 展示。"""
        lines = self.logs()[-limit:]
        return "```\n" + ("\n".join(lines) if lines else "等待训练启动…") + "\n```"

    def render_status(self) -> str:
        """把状态渲染成 Markdown 面板。"""
        status = self.status()
        if status.get("error"):
            return f"### ❌ 训练失败\n\n```\n{status['error']}\n```"
        progress = 0.0
        if status.get("total_steps"):
            progress = 100.0 * status.get("step", 0) / status["total_steps"]
        lines = [
            f"**状态**：{'运行中' if status.get('running') else ('已完成' if status.get('finished') else '未开始')}",
            f"**进度**：epoch {status.get('epoch') or 0}/{status.get('total_epochs') or self.cfg.epochs}"
            f" · step {status.get('step', 0)}/{status.get('total_steps', 0)}（{progress:.1f}%）",
            f"**loss**：{status.get('loss')}（本 epoch 均值 {status.get('avg_loss')}）",
            f"**最佳 PSNR**：{status.get('psnr') or '-'}",
            f"**已用时**：{status.get('elapsed_text', '-')} · **预计剩余**：{status.get('eta_text', '-')}",
            f"**空闲显存**：{status.get('free_vram_gb') or '-'} GB",
            f"**最新消息**：{status.get('message', '-')}",
        ]
        if status.get("output"):
            lines.append(f"**导出文件**：`{status['output']}`")
        return "\n\n".join(lines)


def _tag(image, text: str, height: int = 22):
    """给预览图加一条英文标签栏（OpenCV 不支持中文字形）。"""
    import cv2
    import numpy as np

    bar = np.zeros((height, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (6, int(height * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, image])


def preview_pairs(inputs, *, patch_size: int = 128, scale: int = 4, count: int = 4,
                  seed: int = 0, tag: str = "preview") -> list[str]:
    """生成几张「LR 退化图」预览，让用户在开训前确认退化强度是否合适。

    Returns:
        保存到 ``logs/validate/<tag>/pair_*.png`` 的图片路径列表。
    """
    if not ds.HAS_CV2 or not ds.HAS_NUMPY:
        raise TrainingError("预览需要 opencv-python 与 numpy。")
    files = ds.collect_hr_files(inputs)
    if not files:
        raise TrainingError("没有找到可预览的图片。")
    import random as _random

    import cv2
    import numpy as np

    out_dir = val_mod.validate_root(tag)
    cfg = ds.DegradationConfig(scale=scale)
    paths: list[str] = []
    wanted = max(1, int(count))
    # 逐个尝试，解码失败的图片直接跳过，直到凑够想要的数量
    for path in files:
        if len(paths) >= wanted:
            break
        index = len(paths)
        hr_full = ds.read_image_rgb(path)
        if hr_full is None:
            continue
        size = min(int(patch_size), hr_full.shape[0], hr_full.shape[1])
        rng = _random.Random(seed + index)
        top = rng.randint(0, max(0, hr_full.shape[0] - size))
        left = rng.randint(0, max(0, hr_full.shape[1] - size))
        hr = np.ascontiguousarray(hr_full[top:top + size, left:left + size])
        lr = ds.degrade_image(hr, cfg, _random.Random(seed + index))
        # 左：原始 HR 补丁；右：退化后的 LR 用最近邻放大，直观看出丢了多少细节
        lr_big = cv2.resize(lr, (size, size), interpolation=cv2.INTER_NEAREST)
        panel = np.hstack([_tag(hr, "HR source"), _tag(lr_big, f"LR after degradation (x{scale})")])
        target = out_dir / f"pair_{index:02d}.png"
        cv2.imwrite(str(target), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        paths.append(str(target))
    if not paths:
        raise TrainingError("预览生成失败：所有图片都无法解码。")
    return paths


def default_resume_candidates() -> list[str]:
    """列出可续训的检查点（``*.state.pt``），供 UI 下拉。"""
    if not config.LOG_DIR.exists():
        return []
    files = sorted(config.LORA_DIR.glob("*.state.pt")) + sorted(config.LOG_DIR.rglob("*.state.pt"))
    return ["(不续训)"] + [str(p) for p in files]


__all__ = [
    "HAS_TORCH",
    "HAS_LPIPS",
    "TrainingError",
    "TrainConfig",
    "Trainer",
    "TrainingRunner",
    "estimate_training_vram_gb",
    "preview_pairs",
    "default_resume_candidates",
]
