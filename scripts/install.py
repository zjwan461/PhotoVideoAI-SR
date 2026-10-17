#!/usr/bin/env python3
"""PhotoVideoAI-SR 一键安装脚本（Windows / Linux / macOS）。

把「装依赖」里最容易踩的三个坑按正确顺序处理掉：

1. **PyTorch 必须是 CUDA 版**。默认 PyPI 在 Windows 上给的是 CPU 轮子，
   装完 ``torch.cuda.is_available()`` 为 False，程序会拒绝启动。
2. **basicsr 不能直接装**。``basicsr==1.4.2`` 的元数据声明了 ``tb-nightly``，
   而该包在 PyPI 上已不存在，所以要用 ``--no-deps`` 安装，再手动补齐它
   真正需要的运行时依赖。
3. **basicsr 是源码包**。pip 的隔离构建环境会改用默认源重新联网装构建依赖，
   Windows 下常卡在这一步十几分钟甚至失败，因此加 ``--no-build-isolation``。

用法::

    python scripts/install.py                 # PyTorch cu126（官方源）
    python scripts/install.py --mirror cn     # 国内镜像（PyPI 清华 + PyTorch SJTU）
    python scripts/install.py --skip-torch    # 已装好 CUDA 版 PyTorch，只补其余依赖
    python scripts/install.py --cpu           # 装 CPU 版 PyTorch（仅用于看界面调试）
    python scripts/install.py --dry-run       # 只打印将要执行的命令

脚本是幂等的：已经满足的步骤会跳过，可放心重复运行。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: PyTorch 官方 CUDA 轮子索引（cu126 与当前主流驱动兼容）
TORCH_INDEX_OFFICIAL = "https://download.pytorch.org/whl/cu126"
#: 国内镜像（实测比官方源快一个数量级以上）
TORCH_INDEX_CN = "https://mirror.sjtu.edu.cn/pytorch-wheels/cu126"
TORCH_INDEX_CPU = "https://download.pytorch.org/whl/cpu"
#: 国内 PyPI 镜像
PYPI_CN = "https://pypi.tuna.tsinghua.edu.cn/simple"

#: basicsr 真正需要的运行时依赖（--no-deps 安装后手动补齐）
BASICSR_RUNTIME_DEPS = [
    "addict",
    "future",
    "lmdb",
    "pyyaml",
    "requests",
    "scikit-image",
    "scipy",
    "yapf",
    "facexlib",
]

PIP_BASE = [sys.executable, "-m", "pip", "install", "--no-input", "--progress-bar", "off"]


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def run(cmd: list[str], *, dry_run: bool = False) -> int:
    printable = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    print(f"  $ {printable}")
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def pip_install(
    args: list[str],
    *,
    index: str | None = None,
    dry_run: bool = False,
) -> int:
    cmd = [*PIP_BASE, *args]
    if index:
        cmd += ["-i", index]
    return run(cmd, dry_run=dry_run)


def python_check(code: str) -> tuple[bool, str]:
    """在子进程里跑一段 Python 代码，避免本进程的 import 缓存干扰。"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    output = (completed.stdout or "").strip() or (completed.stderr or "").strip()
    return completed.returncode == 0, output


def torch_state() -> dict | None:
    """返回已安装 PyTorch 的状态；未安装返回 None。"""
    ok, output = python_check(
        "import torch, json;"
        "print(json.dumps({"
        "'version': torch.__version__,"
        "'cuda': torch.version.cuda,"
        "'available': torch.cuda.is_available(),"
        "'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None"
        "}))"
    )
    if not ok:
        return None
    try:
        import json

        return json.loads(output.splitlines()[-1])
    except Exception:  # noqa: BLE001
        return None


def check_python() -> bool:
    section("检查 Python 环境")
    print(f"  Python: {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info < (3, 10):
        print("  ❌ 需要 Python 3.10 或更高版本。")
        return False
    if "WindowsApps" in sys.executable:
        print(
            "  ❌ 当前用的是 Microsoft Store 的 Python 占位程序，它无法正常安装依赖。\n"
            "     请改用 python.org 的安装包或 conda 环境。"
        )
        return False
    print("  ✅ Python 版本满足要求")
    return True


def install_torch(args: argparse.Namespace) -> bool:
    section("安装 CUDA 版 PyTorch")

    state = torch_state()
    if state and args.cpu:
        print(f"  已安装 torch {state['version']}，--cpu 模式下跳过。")
        return True
    if state and state.get("cuda") and state.get("available"):
        print(f"  ✅ 已有可用 CUDA 版 torch {state['version']}（CUDA {state['cuda']}，{state['device']}），跳过。")
        return True
    if state and state.get("cuda") and not state.get("available"):
        print(
            f"  ⚠️ 已装 torch {state['version']}（CUDA {state['cuda']}）但当前不可用；\n"
            "     通常是显卡驱动版本过低。请先更新驱动，再重新运行本脚本。"
        )
        return False
    if state and not state.get("cuda"):
        print(f"  ⚠️ 已装的是 CPU 版 torch {state['version']}，将重装 CUDA 版。")

    if args.cpu:
        index = TORCH_INDEX_CPU
    else:
        index = args.torch_index or (TORCH_INDEX_CN if args.mirror == "cn" else TORCH_INDEX_OFFICIAL)

    code = pip_install(["torch", "torchvision"], index=index, dry_run=args.dry_run)
    if code != 0:
        print("  ❌ PyTorch 安装失败。可尝试手动执行：")
        print(f"     {sys.executable} -m pip install torch torchvision --index-url {index}")
        return False
    if args.dry_run:
        return True

    state = torch_state()
    if state and state.get("available"):
        print(f"  ✅ torch {state['version']}，CUDA {state['cuda']}，设备：{state['device']}")
        return True
    if state and args.cpu:
        print(f"  ✅ torch {state['version']}（CPU 版，仅能用于界面调试）")
        return True
    print("  ❌ PyTorch 已安装但 CUDA 不可用，请检查 NVIDIA 驱动。")
    return False


def install_requirements(args: argparse.Namespace) -> bool:
    section("安装核心依赖（requirements.txt）")
    index = PYPI_CN if args.mirror == "cn" else None
    code = pip_install(
        ["--upgrade", "pip", "setuptools", "wheel"],
        index=index,
        dry_run=args.dry_run,
    )
    if code != 0:
        print("  ⚠️ pip / setuptools / wheel 升级失败，继续尝试安装依赖。")

    code = pip_install(
        ["-r", str(PROJECT_ROOT / "requirements.txt")],
        index=index,
        dry_run=args.dry_run,
    )
    if code != 0:
        print("  ❌ 核心依赖安装失败，请查看上方 pip 输出。")
        return False
    return True


def install_super_resolution(args: argparse.Namespace) -> bool:
    section("安装超分模型栈（basicsr / realesrgan）")
    index = PYPI_CN if args.mirror == "cn" else None

    print("  1/3 补齐 basicsr 的运行时依赖 …")
    if pip_install(BASICSR_RUNTIME_DEPS, index=index, dry_run=args.dry_run) != 0:
        print("  ❌ basicsr 的运行时依赖安装失败。")
        return False

    print("  2/3 安装 basicsr（跳过不可用的 tb-nightly，禁用隔离构建）…")
    if pip_install(
        ["--no-build-isolation", "--no-deps", "basicsr==1.4.2"],
        index=index,
        dry_run=args.dry_run,
    ) != 0:
        print("  ❌ basicsr 安装失败。")
        return False

    print("  3/3 安装 realesrgan …")
    if pip_install(["--no-deps", "realesrgan==0.3.0"], index=index, dry_run=args.dry_run) != 0:
        print("  ❌ realesrgan 安装失败。")
        return False
    return True


def verify(args: argparse.Namespace) -> bool:
    section("验证安装结果")
    if args.dry_run:
        print("  （dry-run，跳过验证）")
        return True

    ok, output = python_check(
        "import sys\n"
        "import numpy, cv2, gradio, torch\n"
        "print(f'numpy={numpy.__version__} opencv={cv2.__version__} gradio={gradio.__version__}')\n"
        "print(f'torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}')\n"
        "import src.img_sr as m\n"
        "print(f'HAS_REALESRGAN={m.HAS_REALESRGAN}')\n"
        "assert m.HAS_REALESRGAN, m.REALESRGAN_IMPORT_ERROR\n"
    )
    print("\n".join(f"  {line}" for line in output.splitlines()))
    if not ok:
        print("  ❌ 验证失败：上面的报错说明了缺失或冲突的依赖。")
        return False
    print("  ✅ 依赖安装完成")
    return True


def next_steps() -> None:
    section("下一步")
    print("  1) 放置基础模型（或让程序首次推理时自动下载）：")
    print(f"     {PROJECT_ROOT / 'model_weights' / 'RealESRGAN_x4plus.pth'}")
    print("  2) 启动 WebUI：")
    print(f"     {sys.executable} app.py")
    print("  3) 自测：")
    print(f"     {sys.executable} tests/selftest.py")
    print(f"     {sys.executable} tests/selftest_e2e.py")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PhotoVideoAI-SR 一键安装脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mirror", choices=["none", "cn"], default="none",
                        help="cn = 使用国内镜像（PyPI 清华 + PyTorch SJTU）")
    parser.add_argument("--torch-index", default=None,
                        help="自定义 PyTorch 轮子索引（默认 cu126）")
    parser.add_argument("--skip-torch", action="store_true",
                        help="跳过 PyTorch 安装（已装好 CUDA 版时使用）")
    parser.add_argument("--cpu", action="store_true",
                        help="安装 CPU 版 PyTorch（仅用于界面调试，推理会失败）")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("PhotoVideoAI-SR 安装脚本")
    print(f"项目目录：{PROJECT_ROOT}")

    if not check_python():
        return 1

    if args.skip_torch:
        section("安装 CUDA 版 PyTorch")
        state = torch_state()
        if state and state.get("available"):
            print(f"  ✅ 已跳过（torch {state['version']}，{state['device']}）")
        else:
            print("  ⚠️ 已按参数跳过；但当前没有可用的 CUDA 版 PyTorch，推理会失败。")
    else:
        if not install_torch(args):
            return 1

    if not install_requirements(args):
        return 1
    if not install_super_resolution(args):
        return 1
    if not verify(args):
        return 1

    next_steps()
    print("\n完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
