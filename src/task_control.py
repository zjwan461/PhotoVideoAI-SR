"""任务取消控制：让「⏹ 取消任务」按钮能真正打断长任务。

为什么需要单独一层
------------------
Gradio 的按钮处理器在队列线程里**同步阻塞**执行，点「取消」只会把另一个事件排进
队列，无法直接掐断正在跑的 Python 调用。因此这里采用协作式取消：

1. 任务开始时用 :func:`task_scope` 注册一个 :class:`CancellationToken`；
2. 取消按钮调用 :func:`cancel_all`，把令牌全部置位；
3. 长任务在**自然边界**（每张图片 / 每一帧 / 模型加载完成后）调用
   :meth:`CancellationToken.raise_if_cancelled`，被置位就抛 :class:`TaskCancelled`；
4. 异常一路抛到 ``app.safe_handler``，转成一句「任务已取消」的提示；
   沿途的 ``try/finally`` 仍会照常清理临时文件，不会留下垃圾。

取消粒度（诚实说明）
--------------------
* **批量图片**：两张图之间生效（单张图片的推理是不可打断的一次调用）；
* **视频逐帧**：每一帧之间检查，通常不到 1 秒就能响应；
* **单张图片 / 首次加载模型**：会在开始推理前检查，所以「正在下载模型」也能被取消。

同一个进程里可以同时有多个任务，:func:`cancel_all` 会一起取消——这与界面上只有一个
取消按钮的行为一致。
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from src.common import get_logger

logger = get_logger("task")


class TaskCancelled(RuntimeError):
    """任务被用户取消。``message`` 已是面向用户的文案，可直接展示。"""

    def __init__(self, message: str = "⏹ 任务已取消") -> None:
        super().__init__(message)


class CancellationToken:
    """单个任务的取消令牌。

    线程安全：任务线程读、界面线程写，底层用 ``threading.Event``。
    """

    __slots__ = ("label", "_event", "_reason")

    def __init__(self, label: str = "任务") -> None:
        self.label = label
        self._event = threading.Event()
        self._reason = ""

    def cancel(self, reason: str = "用户取消") -> None:
        """置位取消标记（幂等）。"""
        self._reason = reason
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def raise_if_cancelled(self) -> None:
        """被取消时抛 :class:`TaskCancelled`；否则立即返回（只是一次 Event 查询）。"""
        if self._event.is_set():
            raise TaskCancelled(f"⏹ {self.label}已取消")

    def wait(self, timeout: float | None = None) -> bool:
        """阻塞等待取消信号（供后台线程使用），返回是否已被取消。"""
        return self._event.wait(timeout)


_LOCK = threading.Lock()
_ACTIVE: dict[int, CancellationToken] = {}


@contextmanager
def task_scope(label: str = "任务") -> Iterator[CancellationToken]:
    """注册一个可被 :func:`cancel_all` 取消的任务作用域。

    Args:
        label: 展示在界面/日志里的任务名，如「图片超分」「视频超分」。
    """
    token = CancellationToken(label)
    with _LOCK:
        _ACTIVE[id(token)] = token
        running = len(_ACTIVE)
    logger.info("任务开始：%s（运行中 %d 个）", label, running)
    try:
        yield token
    finally:
        with _LOCK:
            _ACTIVE.pop(id(token), None)
            running = len(_ACTIVE)
        logger.info("任务结束：%s（运行中 %d 个）", label, running)


def cancel_all(reason: str = "用户取消") -> int:
    """给所有运行中的任务发取消信号，返回受影响的任务数。"""
    with _LOCK:
        tokens = list(_ACTIVE.values())
    for token in tokens:
        token.cancel(reason)
    if tokens:
        logger.warning(
            "已请求取消 %d 个任务：%s", len(tokens), "、".join(t.label for t in tokens)
        )
    return len(tokens)


def active_labels() -> list[str]:
    """当前运行中的任务名列表（供界面提示）。"""
    with _LOCK:
        return [token.label for token in _ACTIVE.values()]


def active_count() -> int:
    """当前运行中的任务数。"""
    with _LOCK:
        return len(_ACTIVE)


__all__ = [
    "TaskCancelled",
    "CancellationToken",
    "task_scope",
    "cancel_all",
    "active_labels",
    "active_count",
]
