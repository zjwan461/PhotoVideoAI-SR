#!/usr/bin/env python3
"""微调模块自测（P2）：不依赖 GPU，用合成数据 + 微型网络验证整条训练链路。

覆盖内容：
1. 退化管线（尺寸、值域、可复现性）
2. 数据集（HR 读取、坏图过滤、过小过滤、配对张量形状）
3. 模型（LoRA 注入、只有 LoRA 可训练、参数量占比）
4. 训练循环（loss 下降、进度回调、验证预览图 + PSNR、checkpoint、断点续训）
5. 导出（只含 LoRA、可被干净模型加载复现、合并进基础权重）

用法：
    python tests/selftest_train.py            # 需要已安装 torch（CPU 亦可）
退出码 0 表示全部通过。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 训练相关的产物统一写到临时目录，避免污染仓库
_SANDBOX = Path(tempfile.mkdtemp(prefix="pvsr_train_selftest_"))
os.environ["PVSR_LORA_DIR"] = str(_SANDBOX / "lora_weights")
os.environ["PVSR_LOG_DIR"] = str(_SANDBOX / "logs")
os.environ["PVSR_MODEL_DIR"] = str(_SANDBOX / "model_weights")

import config  # noqa: E402  (必须在设置环境变量之后导入)
from src.train import dataset as ds  # noqa: E402
from src.train import lora_network as ln  # noqa: E402
from src.train import lora_utils  # noqa: E402
from src.train import trainer as tr
from src.train import validator as vd  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  ✅ {name}")
    else:
        FAILED.append(f"{name} {detail}".strip())
        print(f"  ❌ {name} {detail}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def _make_dataset(directory: Path, count: int = 6, size: int = 96) -> list[Path]:
    """生成合成 HR 图片（含渐变 + 随机纹理，便于退化管线工作）。"""
    import cv2
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    paths: list[Path] = []
    for index in range(count):
        base = np.zeros((size, size, 3), dtype=np.uint8)
        base[:, :, 0] = np.linspace(20, 235, size).astype(np.uint8)[None, :]
        base[:, :, 1] = np.linspace(120, 40, size).astype(np.uint8)[:, None]
        base[:, :, 2] = (rng.random((size, size)) * 255).astype(np.uint8)
        for _ in range(24):
            x, y = rng.integers(0, size - 12, 2)
            cv2.circle(base, (int(x), int(y)), int(rng.integers(2, 7)), (255, 255, 255), -1)
        path = directory / f"hr_{index:02d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(base, cv2.COLOR_RGB2BGR))
        paths.append(path)
    return paths


def test_degradation_and_dataset() -> None:
    section("退化管线与数据集")
    if not ds.HAS_TORCH or not ds.HAS_CV2:
        check("依赖可用", False, "需要 torch + opencv")
        return

    import numpy as np

    hr = (np.random.default_rng(3).random((64, 64, 3)) * 255).astype("uint8")
    lr = ds.degrade_image(hr, ds.DegradationConfig(scale=4), __import__("random").Random(0))
    check("退化尺寸 = 1/scale", lr.shape == (16, 16, 3), str(lr.shape))
    check("退化值域为 uint8", lr.dtype == np.uint8)

    lr_again = ds.degrade_image(hr, ds.DegradationConfig(scale=4), __import__("random").Random(0))
    check("同种子退化可复现", np.array_equal(lr, lr_again))

    data_dir = _SANDBOX / "train_data"
    files = _make_dataset(data_dir, count=6, size=96)
    check("合成数据生成", len(files) == 6)

    # 坏图与过小图必须被过滤
    (data_dir / "broken.png").write_bytes(b"not an image")
    import cv2

    cv2.imwrite(str(data_dir / "tiny.png"), np.zeros((12, 12, 3), np.uint8))

    train_set, val_set, summary = ds.build_dataset(data_dir, patch_size=48, val_ratio=0.34, seed=1)
    check("坏图/过小图被过滤", len(summary["skipped"]) >= 2, f"skipped={summary['skipped']}")
    check("训练/验证切分", summary["train_files"] >= 1 and summary["val_files"] >= 1,
          f"{summary['train_files']}/{summary['val_files']}")

    lr_t, hr_t = train_set[0]
    check("样本张量形状", tuple(lr_t.shape) == (3, 12, 12) and tuple(hr_t.shape) == (3, 48, 48),
          f"{tuple(lr_t.shape)} / {tuple(hr_t.shape)}")
    check("样本值域 [0,1]", float(hr_t.max()) <= 1.0 and float(hr_t.min()) >= 0.0)

    loader = ds.build_dataloader(train_set, batch_size=2, num_workers=0, drop_last=False)
    lr_batch, hr_batch = next(iter(loader))
    check("DataLoader 出 batch", lr_batch.shape[0] <= 2 and hr_batch.shape[1] == 3)


def test_lora_network() -> None:
    section("LoRA 网络层")
    if not ln.HAS_TORCH:
        check("torch 可用", False)
        return

    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet

    net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=16, num_block=1, num_grow_ch=8, scale=4)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        before = net(x).clone()

    injected = ln.inject_lora(net, rank=4, alpha=4.0)
    with torch.no_grad():
        after = net(x)
    check("注入层数 > 0", len(injected) > 0, f"{len(injected)} 层")
    check("注入瞬间输出不变", torch.allclose(before, after, atol=1e-5))

    lora_count, total = ln.mark_only_lora_as_trainable(net, verbosity=False)
    trainable = [n for n, p in net.named_parameters() if p.requires_grad]
    check("只有 LoRA 可训练", all(".lora_" in n for n in trainable), f"{len(trainable)} 个张量")
    check("LoRA 参数量远小于总量", lora_count < total * 0.5, f"{lora_count}/{total}")

    summary = ln.lora_summary(net)
    check("结构摘要可用", summary["layers"] == len(injected) and summary["rank"] == 4, str(summary))

    # 合并 / 反合并
    merged = ln.merge_lora_to_base(net)
    unmerged = ln.unmerge_lora_from_base(net)
    with torch.no_grad():
        restored = net(x)
    check("合并/反合并可逆", merged == len(injected) and unmerged == len(injected)
          and torch.allclose(restored, after, atol=1e-4))

    ln.set_lora_scaling(net, 0.0)
    with torch.no_grad():
        disabled = net(x)
    check("scaling=0 等价于关闭 LoRA", torch.allclose(disabled, before, atol=1e-5))


def test_training_loop() -> None:
    section("训练循环 / 验证 / 断点续训")
    if not tr.HAS_TORCH:
        check("torch 可用", False)
        return

    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet

    data_dir = _SANDBOX / "train_data"
    factory = lambda _name: RRDBNet(  # noqa: E731
        num_in_ch=3, num_out_ch=3, num_feat=16, num_block=1, num_grow_ch=8, scale=4
    )

    cfg = tr.TrainConfig(
        dataset_paths=[str(data_dir)],
        lora_name="selftest_lora",
        epochs=2,
        batch_size=2,
        patch_size=48,
        learning_rate=1e-3,
        rank=4,
        val_ratio=0.34,
        val_interval=1,
        val_images=2,
        save_interval=1,
        keep_last_checkpoints=2,
        device="cpu",
        num_workers=0,
        seed=0,
        log_every=1,
    )
    logs: list[str] = []
    trainer = tr.Trainer(cfg, log_cb=logs.append, network_factory=factory)
    result = trainer.run()

    check("训练完成并导出 LoRA", Path(result["output_path"]).is_file(), result["output_path"])
    check("导出文件只含 LoRA（体积小）", result["final_size_mb"] < 20, f"{result['final_size_mb']} MB")
    check("续训状态文件已保存", result["state_path"] and Path(result["state_path"]).is_file())
    check("续训状态不放进 lora_weights/",
          Path(result["state_path"]).parent != Path(config.LORA_DIR),
          str(result["state_path"]))
    check("续训状态不出现在 LoRA 下拉", all(not n.endswith(".state.pt") for n in
                                          __import__("src.common", fromlist=["scan_lora_weights"]).scan_lora_weights()))
    check("验证 PSNR 有记录", result["best_psnr"] is not None, str(result["best_psnr"]))
    check("loss 曲线非空", len(result["history"]) == 2, str(result["history"]))
    check("训练摘要写入 logs", (Path(config.LOG_DIR) / "train" / cfg.run_tag / "summary.json").is_file())

    images = vd.list_validation_images(cfg.run_tag, limit=10)
    check("验证预览图已生成", len(images) >= 1, f"{len(images)} 张")

    payload = torch.load(result["output_path"], map_location="cpu", weights_only=False)
    check("导出格式带 format 标记", payload.get("format") == "photovideoai-sr-lora")
    check("导出只包含 LoRA 张量",
          all(".lora_" in k for k in payload["lora_state_dict"]),
          str(list(payload["lora_state_dict"])[:2]))

    # 干净模型 + LoRA 复现训练结果
    clean = factory(cfg.base_model)
    info = lora_utils.attach_lora_for_inference(clean, result["output_path"])
    check("干净模型可加载该 LoRA", info["loaded"] > 0, str(info)[:80])

    # 断点续训：从 state.pt 恢复，epoch 应从 2 继续
    resume_cfg = tr.TrainConfig(**{**cfg.to_dict(), "epochs": 3, "resume_from": result["state_path"]})
    resume_trainer = tr.Trainer(resume_cfg, log_cb=logs.append, network_factory=factory)
    resume_result = resume_trainer.run()
    check("断点续训成功", Path(resume_result["output_path"]).is_file())
    check("续训从上次 epoch 继续", resume_trainer.start_epoch == 2, str(resume_trainer.start_epoch))

    # 早期停止
    cfg_stop = tr.TrainConfig(**{**cfg.to_dict(), "lora_name": "selftest_stop", "epochs": 5})
    stop_trainer = tr.Trainer(cfg_stop, log_cb=logs.append, network_factory=factory,
                              should_stop=lambda: True)
    stop_result = stop_trainer.run()
    check("停止请求能被响应并保存", Path(stop_result["output_path"]).is_file())

    check("日志回调有输出", len([line for line in logs if "训练" in line]) > 0, f"{len(logs)} 行")


def test_runner_and_errors() -> None:
    section("后台运行器与错误处理")
    if not tr.HAS_TORCH:
        return
    import random

    import numpy as np
    from basicsr.archs.rrdbnet_arch import RRDBNet

    data_dir = _SANDBOX / "train_data"
    factory = lambda _name: RRDBNet(  # noqa: E731
        num_in_ch=3, num_out_ch=3, num_feat=16, num_block=1, num_grow_ch=8, scale=4
    )
    cfg = tr.TrainConfig(
        dataset_paths=[str(data_dir)], lora_name="runner_lora", epochs=1,
        batch_size=2, patch_size=48, rank=4, val_ratio=0.2, val_interval=1,
        save_interval=1, device="cpu", seed=0,
    )
    runner = tr.TrainingRunner(cfg, network_factory=factory)
    runner.start()
    runner.join(timeout=600)
    status = runner.status()
    check("运行器完成并给出状态", bool(status.get("finished")), str(status.get("error")))
    check("运行器日志非空", len(runner.logs()) > 0)
    check("运行器渲染状态", "状态" in runner.render_status())
    check("运行器结果含输出路径", bool(runner.result and runner.result.get("output_path")))
    runner.stop()  # 已结束，重复调用应安全

    # 显存建议文案（模拟 CUDA OOM 异常 → 可读提示）
    from src.common import friendly_cuda_error

    oom_text = friendly_cuda_error(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    check("OOM 提示友好化", oom_text is not None and "显存" in oom_text)

    # 退化预览
    try:
        preview_paths = tr.preview_pairs([str(data_dir)], patch_size=48, scale=4, count=2, seed=0)
        check("退化预览生成", len(preview_paths) == 2, str(preview_paths))
    except Exception as exc:  # noqa: BLE001
        check("退化预览生成", False, str(exc)[:80])

    # 无效配置必须报错
    try:
        tr.TrainConfig(dataset_paths=[], lora_name="x").validate()
        check("空数据集校验", False)
    except tr.TrainingError:
        check("空数据集校验", True)

    try:
        tr.TrainConfig(dataset_paths=[str(data_dir)], patch_size=8).validate()
        check("过小 patch 校验", False)
    except tr.TrainingError:
        check("过小 patch 校验", True)

    # 生成器随机性检查（退化预览不应每次完全一样）
    random.seed(0)
    hr = (np.random.default_rng(1).random((48, 48, 3)) * 255).astype("uint8")
    a = ds.degrade_image(hr, ds.DegradationConfig(scale=4), random.Random(1))
    b = ds.degrade_image(hr, ds.DegradationConfig(scale=4), random.Random(2))
    check("不同种子退化结果不同", not np.array_equal(a, b))


def main() -> int:
    print("PhotoVideoAI-SR 微调模块自测开始")
    print(f"沙箱目录：{_SANDBOX}")
    if not tr.HAS_TORCH:
        print("未安装 PyTorch，仅执行退化/数据集静态检查。")
    try:
        test_degradation_and_dataset()
        test_lora_network()
        test_training_loop()
        test_runner_and_errors()
    finally:
        print("\n" + "=" * 60)
        print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
        for item in FAILED:
            print(f"  - 失败：{item}")
        if not FAILED and os.environ.get("PVSR_KEEP_SELFTEST", "0") not in ("1", "true", "True"):
            shutil.rmtree(_SANDBOX, ignore_errors=True)
            print(f"已清理沙箱目录 {_SANDBOX}")
        else:
            print(f"保留沙箱目录供排查：{_SANDBOX}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
