"""第三方依赖兼容补丁：让老库能在新版 torchvision 上正常导入。

背景
----
``basicsr==1.4.2``（以及 ``gfpgan`` 老版本）在模块顶层写的是::

    from torchvision.transforms.functional_tensor import rgb_to_grayscale

而 ``torchvision.transforms.functional_tensor`` 这个子模块自 **torchvision 0.17**
起被删除，等价函数迁到了 ``torchvision.transforms.functional``。

这跟 numpy 版本无关。只要 torchvision 够新，``import realesrgan`` 就会失败：
``realesrgan`` 的 ``models`` / ``data`` 包会连带
``import basicsr.data.degradations``，而该文件第 8 行正是上面那条导入。
表现出来就是一个 ``ModuleNotFoundError``，很容易被误判成「没装 basicsr」。

处理方式
--------
在 ``sys.modules`` 里注册一个**只做转发的别名模块**，让老代码的导入语句继续成立。
这样做的优点：

* 不修改 site-packages，不复制第三方源码，重装依赖后依然有效；
* 不依赖导入顺序：由本项目在导入 basicsr / realesrgan 之前显式调用一次即可；
* 幂等，重复调用无副作用。

用法
----
在导入 basicsr / realesrgan / gfpgan **之前**调用一次::

    from src.torchvision_compat import ensure_functional_tensor

    ensure_functional_tensor()
    from realesrgan import RealESRGANer
"""

from __future__ import annotations

import sys
import types

#: 被移除的子模块名（也是老库导入时写死的路径）
_SUBMODULE = "torchvision.transforms.functional_tensor"

#: 垫片模块的说明文档
_SHIM_DOC = (
    "兼容垫片（由 PhotoVideoAI-SR 注册）："
    "转发到 torchvision.transforms.functional，函数实现仍来自 torchvision。"
)


def ensure_functional_tensor() -> str:
    """确保 ``torchvision.transforms.functional_tensor`` 可被导入。

    幂等：已经可用或已注册过垫片时直接返回。

    Returns:
        空串表示已就绪（torchvision 原生存在，或垫片注册成功）；
        非空字符串为失败原因，调用方可将其拼进面向用户的可读错误信息。
    """
    if _SUBMODULE in sys.modules:
        return ""
    try:  # 老版本 torchvision 原生就有，直接用，不做任何改动
        import torchvision.transforms.functional_tensor  # noqa: F401

        return ""
    except Exception:  # noqa: BLE001 - 预期内的缺失，继续装垫片
        pass

    try:
        import torchvision.transforms as transforms

        # 优先复用真正的 legacy 实现：torchvision 仍以私有名保留该模块。
        # 拿不到时退回公开的 functional 模块（内含同名 rgb_to_grayscale 等函数）。
        source = getattr(transforms, "_functional_tensor", None)
        if source is None:
            source = transforms.functional

        shim = types.ModuleType(_SUBMODULE)
        shim.__doc__ = _SHIM_DOC
        for key, value in vars(source).items():
            if not key.startswith("__"):
                setattr(shim, key, value)

        sys.modules[_SUBMODULE] = shim
        # 同时挂到父模块属性上，兼容 `torchvision.transforms.functional_tensor.xxx` 写法
        setattr(transforms, "functional_tensor", shim)
        return ""
    except Exception as exc:  # noqa: BLE001
        return str(exc)


__all__ = ["ensure_functional_tensor"]
