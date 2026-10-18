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
    python scripts/install.py --maxine        # 额外安装 Maxine VSR（nvidia-vfx，约 400 MB）
    python scripts/install.py --gfpgan        # 额外安装 GFPGAN 人脸修复（库 + 约 332 MB 权重）
    python scripts/install.py --restore       # 额外下载前置复原模型（SwinIR，约 313 MB）
    python scripts/install.py --all           # 以上全部 + lpips（共约 1.2 GB）
    python scripts/install.py --dry-run       # 只打印将要执行的命令

脚本是幂等的：已经满足的步骤会跳过，可放心重复运行。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: PyTorch 官方 CUDA 轮子索引（cu126 与当前主流驱动兼容）
TORCH_INDEX_OFFICIAL = "https://download.pytorch.org/whl/cu126"
#: 国内镜像（实测比官方源快一个数量级以上）
TORCH_INDEX_CN = "https://mirror.sjtu.edu.cn/pytorch-wheels/cu126"
TORCH_INDEX_CPU = "https://download.pytorch.org/whl/cpu"
#: 国内 PyPI 镜像
PYPI_CN = "https://pypi.tuna.tsinghua.edu.cn/simple"
#: NVIDIA 官方 wheel 索引（Maxine VSR / nvidia-vfx 只在这里发布）。
#: 注意：PyPI 上的 ``nvidia-vfx`` 只是一个 3.6 KB 的占位包，装它得不到 ``nvvfx``。
MAXINE_INDEX = "https://pypi.nvidia.com"

#: GFPGAN 人脸修复权重（官方发布地址）
GFPGAN_WEIGHTS_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
)
#: 权重文件名（必须与 config.gfpgan_weights_path() 一致）
GFPGAN_WEIGHTS_NAME = "GFPGANv1.4.pth"
#: 下载 User-Agent，部分 CDN 会拒绝 urllib 默认值
_UA = "PhotoVideoAI-SR-installer/1.0"

#: Maxine VSR 的硬件/驱动要求（失败时原样打印给用户）
MAXINE_REQUIREMENTS = """\
  * GPU：带 Tensor Core 的 NVIDIA 显卡（Turing / Ampere / Ada / Blackwell / Hopper）
  * 驱动：Windows 570.65+（TCC 设备需 595+）；Linux 570.190+ / 580.82+ / 590.44+
  * Python 3.10+；Windows 10+ 或 Ubuntu 20.04/22.04/24.04、Debian 12、RHEL 8/9
  * wheel 约 400 MB（含 VFX 运行时库），首次安装需要能访问 pypi.nvidia.com"""

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


def install_maxine(args: argparse.Namespace) -> bool:
    """安装 NVIDIA Maxine VSR（``nvidia-vfx``，可选闭源组件）。

    只在显式传入 ``--maxine`` 时执行：wheel 约 400 MB，且对显卡/驱动有要求。
    """
    section("安装 Maxine VSR（可选，闭源组件）")

    ok, _ = python_check("import nvvfx")
    if ok:
        print("  ✅ 已安装 nvidia-vfx（nvvfx 可导入），跳过。")
        return True

    index = args.maxine_index or MAXINE_INDEX
    print("  说明：这是 NVIDIA 闭源组件，不支持 LoRA；装好后视频 Tab 会多出 Maxine 选项。")
    print(MAXINE_REQUIREMENTS)

    code = pip_install(["nvidia-vfx"], index=index, dry_run=args.dry_run)
    if code != 0:
        print("  ❌ nvidia-vfx 安装失败。可手动执行：")
        print(f"     {sys.executable} -m pip install nvidia-vfx --index-url {index}")
        return False
    if args.dry_run:
        return True

    ok, output = python_check(
        "import nvvfx, json;"
        "print(json.dumps({"
        "'version': getattr(nvvfx, '__version__', 'unknown'),"
        "'has_vsr': hasattr(nvvfx, 'VideoSuperRes')}))"
    )
    if not ok:
        print("  ❌ 已安装但无法导入 nvvfx：")
        print("\n".join(f"     {line}" for line in output.splitlines()[:6]))
        print("     常见原因：显卡不带 Tensor Core、驱动版本过低，或缺少 VC++ 运行库。")
        return False

    print(f"  ✅ nvidia-vfx 可用：{output.splitlines()[-1]}")
    return True


def download_file(
    url: str,
    target: Path,
    *,
    timeout: int = 60,
    min_bytes: int = 1,
) -> bool:
    """下载文件到 ``target``：支持断点续传、进度显示与低速放弃。

    只用标准库——这个脚本要在装依赖之前就能跑。
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size >= min_bytes:
        print(f"  已存在，跳过下载：{target.name}（{target.stat().st_size / 1e6:.0f} MB）")
        return True

    part = target.with_name(target.name + ".part")
    offset = part.stat().st_size if part.is_file() else 0
    if offset:
        print(f"  发现未完成的下载，从 {offset / 1e6:.0f} MB 处续传…")

    request = urllib.request.Request(url, headers={"User-Agent": _UA})
    if offset:
        request.add_header("Range", f"bytes={offset}-")

    try:
        response = urllib.request.urlopen(request, timeout=timeout)  # noqa: S310
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and part.is_file():  # 本地已下完，只是没改名
            part.replace(target)
            print(f"  ✅ 已下载：{target}")
            return True
        print(f"  ❌ 下载失败（HTTP {exc.code}）：{url}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ 下载失败：{exc}")
        return False

    with response:
        status = getattr(response, "status", 200)
        if offset and status != 206:
            print("  服务器不支持断点续传，重新下载…")
            offset = 0
            part.unlink(missing_ok=True)
        total = int(response.headers.get("Content-Length") or 0) + offset
        received = offset
        started = time.time()
        with open(part, "ab" if offset else "wb") as handle:
            while True:
                chunk = response.read(1024 * 512)
                if not chunk:
                    break
                handle.write(chunk)
                received += len(chunk)
                elapsed = max(time.time() - started, 1e-6)
                if total:
                    print(
                        f"\r  下载中 {received / 1e6:7.1f} / {total / 1e6:.1f} MB"
                        f"（{(received - offset) / elapsed / 1e6:.1f} MB/s）",
                        end="",
                        flush=True,
                    )
                if elapsed > 30 and (received - offset) / elapsed < 64 * 1024:
                    print("\n  ❌ 速度过低（持续 30 秒低于 64 KB/s），已放弃。")
                    return False
    print()

    if total and received < total:
        print(
            f"  ⚠️ 下载不完整（{received / 1e6:.1f}/{total / 1e6:.1f} MB），"
            f"已保留 {part.name} 供下次续传。"
        )
        return False
    if received < min_bytes:
        print(f"  ❌ 文件过小（{received} 字节），可能地址失效。")
        return False
    part.replace(target)
    print(f"  ✅ 已下载：{target}（{received / 1e6:.0f} MB）")
    return True


def gfpgan_weights_path() -> Path:
    """向项目要 GFPGAN 权重的规范路径（尊重 PVSR_MODEL_DIR 覆盖）。"""
    ok, output = python_check("import config; print(config.gfpgan_weights_path())")
    if ok and output.strip():
        return Path(output.strip().splitlines()[-1])
    return PROJECT_ROOT / "model_weights" / GFPGAN_WEIGHTS_NAME


def install_gfpgan(args: argparse.Namespace) -> bool:
    """安装 GFPGAN 人脸修复：Python 库 + 权重文件（权重约 332 MB）。

    注意 ``gfpgan 1.3.8`` 的元数据同样声明了 ``tb-nightly``（PyPI 上已不存在），
    所以库要用 ``--no-deps`` 安装；它的运行时依赖已由 basicsr 那一步装齐。
    """
    section("安装 GFPGAN 人脸修复（可选）")

    weights = gfpgan_weights_path()
    weights_ok = weights.is_file() and weights.stat().st_size > 100 * 1024 * 1024
    # 裸 `import gfpgan` 会连 basicsr 一起踩 torchvision 已删除子模块的坑，
    # 所以和项目运行时保持一致：先装兼容垫片再导入。
    lib_ok, _ = python_check(
        "from src.torchvision_compat import ensure_functional_tensor;"
        "ensure_functional_tensor();"
        "from gfpgan import GFPGANer;"
        "print('ok')"
    )

    if lib_ok and weights_ok:
        print("  ✅ gfpgan 库与权重都已就绪，跳过。")
        return True

    if not lib_ok:
        print("  说明：gfpgan 1.3.8 声明了已下线的 tb-nightly，需用 --no-deps 安装；")
        print("        它的运行时依赖（basicsr / facexlib / lmdb / scipy 等）前面已装好。")
        index = PYPI_CN if args.mirror == "cn" else None
        if pip_install(["--no-deps", "gfpgan==1.3.8"], index=index, dry_run=args.dry_run) != 0:
            print("  ❌ gfpgan 库安装失败。")
            return False

    if not weights_ok:
        url = args.gfpgan_url or GFPGAN_WEIGHTS_URL
        print(f"  下载权重 {GFPGAN_WEIGHTS_NAME}（约 332 MB）→ {weights}")
        print(f"  来源：{url}")
        if args.dry_run:
            print(f"  $ （dry-run）下载 {url}")
        elif not download_file(url, weights, min_bytes=100 * 1024 * 1024):
            print("  ❌ 权重下载失败。可手动下载后放到：")
            print(f"     {weights}")
            print(f"     地址：{url}")
            return False

    # GFPGAN 还会经 facexlib 加载人脸检测/解析模型；缺它们时首次点击会**静默联网**
    # 长时间无响应（GPU 也不会有占用），所以这里一并下载好。
    ok, output = python_check(
        "import config, json;"
        "print(json.dumps({"
        "'dir': str(config.facexlib_weights_dir() or ''),"
        "'models': config.GFPGAN_FACEXLIB_MODELS,"
        "'missing': config.missing_facexlib_models()}))"
    )
    if not ok:
        print("  ❌ 无法读取 facexlib 模型清单：")
        print("\n".join(f"     {line}" for line in output.splitlines()[:4]))
        return False
    facexlib_info = json.loads(output.strip().splitlines()[-1])
    facexlib_dir = Path(facexlib_info["dir"]) if facexlib_info["dir"] else None
    if facexlib_dir is None:
        print("  ❌ 未找到 facexlib（它是 gfpgan 的依赖），请先完成前面的安装步骤。")
        return False

    pending = dict.fromkeys(facexlib_info["missing"], True)
    if pending:
        print(f"  下载 facexlib 人脸检测/解析模型（{len(pending)} 个，共约 185 MB）")
        print(f"  目标目录：{facexlib_dir}")
    for name in pending:
        url = facexlib_info["models"].get(name, "")
        if not url:
            print(f"  ⚠️ {name} 没有已知下载地址，请手动放入 {facexlib_dir}")
            return False
        if args.dry_run:
            print(f"  $ （dry-run）下载 {url}")
            continue
        print(f"  → {name}")
        if not download_file(url, facexlib_dir / name, min_bytes=20 * 1024 * 1024):
            print(f"  ❌ {name} 下载失败。可手动下载后放到：{facexlib_dir}")
            print(f"     地址：{url}")
            return False

    if args.dry_run:
        return True

    # 真验证一次：权重能被 torch 读出来，且 config 判定为可用
    ok, output = python_check(
        "import config;"
        "p = config.gfpgan_weights_path();"
        "print('path=', p);"
        "print('size_mb=', round(p.stat().st_size / 1e6, 1));"
        "import torch;"
        "d = torch.load(str(p), map_location='cpu', weights_only=False);"
        "print('keys=', list(d)[:3] if isinstance(d, dict) else type(d).__name__);"
        "print('gfpgan_available=', config.gfpgan_available())"
    )
    if not ok or "gfpgan_available= True" not in output:
        print("  ❌ 安装后仍不可用：")
        print("\n".join(f"     {line}" for line in output.splitlines()[:8]))
        return False
    print("\n".join(f"  {line}" for line in output.splitlines()))
    print("  ✅ 人脸修复已可用（图片 Tab 会出现「人脸修复 (GFPGAN)」开关）")
    return True


def probe_command(key: str) -> str:
    """构造复原模型自检的子进程代码。

    单独成函数，测试可以直接执行**同一份**代码——避免这里写成
    ``from src import config``（config 是顶层模块，不在 src 里）这类笔误只能等用户踩到。
    """
    return (
        "import config\n"
        "from src import restore\n"
        f"ok, text = restore.probe('{key}')\n"
        "print('OK' if ok else 'FAIL', text)\n"
    )


def install_restore(args: argparse.Namespace) -> bool:
    """下载老照片前置复原模型的权重（SwinIR，可选）。

    网络结构随 ``basicsr`` 一起提供（``basicsr.archs.swinir_arch``），
    所以这一步**只下载权重**，不需要额外 pip 包。
    """
    section("安装前置复原模型（可选，老照片去压缩块/去噪）")

    ok, output = python_check(
        "import config, json;"
        "print(json.dumps({"
        "'dir': str(config.RESTORE_DIR),"
        "'models': config.RESTORE_MODELS,"
        "'missing': [k for k in config.RESTORE_MODELS if not config.restore_installed(k)]}))"
    )
    if not ok:
        print("  ❌ 无法读取复原模型清单（项目模块导入失败）：")
        print("\n".join(f"     {line}" for line in output.splitlines()[:5]))
        return False

    info = json.loads(output.strip().splitlines()[-1])
    target_dir = Path(info["dir"])
    models = info["models"]
    missing = info["missing"]

    if not missing:
        print("  ✅ 复原模型都已就绪，跳过。")
        return True

    total = sum(models[key]["size_mb"] for key in missing)
    print(f"  需要下载 {len(missing)} 个模型，共约 {total} MB → {target_dir}")
    for key in missing:
        entry = models[key]
        print(f"  → {entry['label']}")
        if args.dry_run:
            print(f"  $ （dry-run）下载 {entry['url']}")
            continue
        if not download_file(
            entry["url"], target_dir / entry["file"], min_bytes=20 * 1024 * 1024
        ):
            print(f"  ❌ {entry['file']} 下载失败。可手动下载后放到：{target_dir}")
            print(f"     地址：{entry['url']}")
            return False

    if args.dry_run:
        return True

    ok, output = python_check(
        "import config;"
        "print('installed=', [k for k in config.RESTORE_MODELS if config.restore_installed(k)]);"
        "print('restore_available=', config.restore_available())"
    )
    print("\n".join(f"  {line}" for line in output.splitlines()))
    if not ok or "restore_available= True" not in output:
        print("  ❌ 复原模型安装后仍不可用。")
        return False

    # 自检：复原模型对平坦输入应近似恒等；配置（img_range/channels）取错会静默失真
    for key in missing:
        ok_probe, text = python_check(probe_command(key))
        line = (text or "").strip().splitlines()[-1] if (text or "").strip() else ""
        if not ok_probe:
            # 自检脚本本身跑不起来（不是权重的问题），不应该据此判定安装失败
            print(f"  ⚠️ 无法执行 {key} 自检：{line[:120]}")
            continue
        if not line.startswith("OK"):
            print(f"  ❌ {key} 自检未通过：{line[3:] if line.startswith('FAIL') else line}")
            print("     该权重与当前架构配置可能不匹配，请不要使用。")
            return False
        print(f"  ✅ {key} 自检通过：{line[3:]}")

    print("  ✅ 前置复原已可用（图片 Tab 会出现「前置复原」下拉）")
    return True


def install_lpips(args: argparse.Namespace) -> bool:
    """安装可选依赖 lpips（LoRA 微调的感知损失）。

    只是装库本身（很小）；它会在**首次使用时**才去下载 VGG 权重（约 500 MB），
    所以这里不会产生额外的大文件。
    """
    section("安装可选依赖 lpips（微调感知损失）")
    ok, _ = python_check("import lpips")
    if ok:
        print("  ✅ 已安装，跳过。")
        return True
    index = PYPI_CN if args.mirror == "cn" else None
    if pip_install(["lpips>=0.1.4"], index=index, dry_run=args.dry_run) != 0:
        print("  ❌ lpips 安装失败（不影响主流程，微调时会退回 L1 损失）。")
        return False
    if args.dry_run:
        return True
    ok, output = python_check("import lpips; print('lpips', lpips.__version__ if hasattr(lpips,'__version__') else 'ok')")
    if not ok:
        print("  ❌ lpips 安装后仍无法导入：")
        print("\n".join(f"     {line}" for line in output.splitlines()[:4]))
        return False
    print("  ✅ lpips 已安装（首次用于微调时会下载 VGG 权重约 500 MB）")
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

    # Maxine 是可选组件，装没装都不影响主流程，只做信息展示
    maxine_ok, maxine_out = python_check(
        "import nvvfx; print(getattr(nvvfx, '__version__', 'unknown'))"
    )
    if maxine_ok:
        print(f"  nvvfx（Maxine VSR）: 可用 {maxine_out.splitlines()[-1]}")
    else:
        print("  nvvfx（Maxine VSR）: 未安装（可选，视频 Tab 不会显示该选项）")

    print("  ✅ 依赖安装完成")
    return True


def next_steps(*, maxine: bool = False, gfpgan: bool = False, restore: bool = False) -> None:
    section("下一步")
    print("  1) 放置基础模型（或让程序首次推理时自动下载）：")
    print(f"     {PROJECT_ROOT / 'model_weights' / 'RealESRGAN_x4plus.pth'}")
    print("  2) 启动 WebUI：")
    print(f"     {sys.executable} app.py")
    print("  3) 自测：")
    print(f"     {sys.executable} tests/selftest.py")
    print(f"     {sys.executable} tests/selftest_e2e.py")
    if not maxine:
        print("  4) 想用 Maxine VSR 时序超分，再执行：")
        print(f"     {sys.executable} scripts/install.py --maxine")
    if not gfpgan:
        print("  5) 想用 GFPGAN 老照片人脸修复，再执行：")
        print(f"     {sys.executable} scripts/install.py --gfpgan")
    if not restore:
        print("  6) 想用老照片前置复原（去压缩块/去噪），再执行：")
        print(f"     {sys.executable} scripts/install.py --restore")


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
    parser.add_argument("--maxine", action="store_true",
                        help="额外安装 NVIDIA Maxine VSR（nvidia-vfx，约 400 MB，可选）")
    parser.add_argument("--maxine-index", default=None,
                        help=f"nvidia-vfx 的索引地址（默认 {MAXINE_INDEX}）")
    parser.add_argument("--gfpgan", action="store_true",
                        help="额外安装 GFPGAN 人脸修复（库 + 约 332 MB 权重，可选）")
    parser.add_argument("--gfpgan-url", default=None,
                        help=f"GFPGAN 权重下载地址（默认官方 GitHub Releases）")
    parser.add_argument("--restore", action="store_true",
                        help="额外下载老照片前置复原模型（SwinIR，约 313 MB，可选）")
    parser.add_argument("--all", action="store_true",
                        help="安装全部可选组件：--maxine --gfpgan --restore，外加 lpips"
                             "（共约 1.2 GB 下载，耗时取决于网速）")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    args = parser.parse_args(argv)
    if args.all:
        # --all 只是把三个开关一起打开，后续逻辑与逐个传参完全一致
        args.maxine = True
        args.gfpgan = True
        args.restore = True
    return args


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
    if args.maxine:
        if not install_maxine(args):
            return 1
    if args.gfpgan:
        if not install_gfpgan(args):
            return 1
    if args.restore:
        if not install_restore(args):
            return 1
    if args.all:
        # lpips 只用于微调的感知损失，装不上也不影响主流程，因此只警告不中断
        if not install_lpips(args):
            print("  ⚠️ lpips 未装上，其余组件不受影响（微调时会自动退回 L1 损失）。")
    if not verify(args):
        return 1

    next_steps(maxine=args.maxine, gfpgan=args.gfpgan, restore=args.restore)
    print("\n完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
