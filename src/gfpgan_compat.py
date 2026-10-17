"""修 GFPGAN / facexlib 的权重目录，让它与「启动目录」无关。

问题
----
``gfpgan/utils.py`` 构造人脸助手时传的是::

    FaceRestoreHelper(..., model_rootpath='gfpgan/weights')

而 facexlib 的 ``load_file_from_url`` 会把这个值**原样**当成 ``save_dir``：

    if save_dir is None:
        save_dir = os.path.join(ROOT_DIR, model_dir)
    os.makedirs(save_dir, exist_ok=True)

于是它相对**当前工作目录**去建 ``gfpgan/weights/`` 并联网下载检测/解析模型。
后果：

* 从别的目录启动 WebUI（例如 ``python E:\\...\\app.py``）时，会在那个目录下重建一份；
* 权重缺失时首次点击「人脸修复」，表现是**长时间没反应**（CPU/GPU 都不动），
  因为阻塞在 GitHub 下载上（国内网络下可达数十分钟）。

处理
----
把相对 ``save_dir`` 统一锚定到项目内的 ``gfpgan/weights/``（绝对路径），
并用同一个目录做「是否齐全」的判断与安装脚本的下载目标，三处保持一致。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import config
from src.common import get_logger

logger = get_logger("gfpgan_compat")

#: 在 facexlib 的模块属性上打的标记，表示已经替换过
_MARKER = "_pvsr_pinned_weights_dir"


def weights_dir() -> Path:
    """GFPGAN / facexlib 权重目录（绝对路径，与工作目录无关）。"""
    return config.facexlib_weights_dir()


def pin_weights_dir(target: Path | None = None) -> Path:
    """把 facexlib 的权重目录钉到项目内，返回该目录。

    幂等：重复调用只会再确认一次路径。facexlib 未安装时安静返回路径，
    由调用方（或可用性判断）去处理缺失情况。
    """
    directory = Path(target or config.facexlib_weights_dir()).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    try:
        import facexlib.utils.misc as misc
    except Exception as exc:  # noqa: BLE001 - 未安装 facexlib 时无需打补丁
        logger.debug("facexlib 不可用，跳过权重目录补丁：%s", exc)
        return directory

    if getattr(misc, _MARKER, False):
        return directory

    original = misc.load_file_from_url

    def load_file_from_url(
        url,
        model_dir=None,
        progress=True,
        file_name=None,
        save_dir=None,
    ):
        # 相对路径（含 gfpgan 传进来的 'gfpgan/weights'）一律改到项目目录
        if save_dir is None or not os.path.isabs(str(save_dir)):
            save_dir = str(directory)
        return original(
            url,
            model_dir=model_dir,
            progress=progress,
            file_name=file_name,
            save_dir=save_dir,
        )

    misc.load_file_from_url = load_file_from_url
    setattr(misc, _MARKER, True)

    # 其它 facexlib 子模块可能已经 `from facexlib.utils import load_file_from_url`
    # 绑定了原函数，这里一并替换，避免漏网。
    for name, module in list(sys.modules.items()):
        if not name.startswith("facexlib") or module is None:
            continue
        if getattr(module, "load_file_from_url", None) is original:
            setattr(module, "load_file_from_url", load_file_from_url)
        setattr(module, _MARKER, True)

    logger.debug("facexlib 权重目录已锚定：%s", directory)
    return directory


__all__ = ["pin_weights_dir", "weights_dir"]
