from __future__ import annotations

import asyncio
import threading
from typing import Any, Awaitable, Optional


_loop: Optional[asyncio.AbstractEventLoop] = None
_thread: Optional[threading.Thread] = None
_lock = threading.Lock()


def _loop_worker(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def ensure_background_loop() -> asyncio.AbstractEventLoop:
    global _loop, _thread
    if _loop and _thread and _thread.is_alive():
        return _loop
    with _lock:
        if _loop and _thread and _thread.is_alive():
            return _loop
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=_loop_worker, args=(loop,), name="agent-bg-loop", daemon=True)
        thread.start()
        _loop = loop
        _thread = thread
        return loop


def run_coro_sync(awaitable: Awaitable[Any], timeout: Optional[float] = None) -> Any:
    """
    在后台事件循环中运行协程，并在当前线程同步等待结果。

    注意：协程自身可实现超时控制；如需外部超时，可传入 timeout。
    """
    loop = ensure_background_loop()
    fut = asyncio.run_coroutine_threadsafe(awaitable, loop)
    return fut.result(timeout=timeout)


