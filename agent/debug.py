from __future__ import annotations

import json
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional
import contextvars


def _safe_serialize(obj: Any, limit: int = 800) -> str:
    try:
        text = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        text = str(obj)
    if isinstance(text, str) and len(text) > limit:
        return text[:limit] + "…(截断)"
    return text


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for k, v in value.items():
            lk = str(k).lower()
            if any(x in lk for x in ["api_key", "apikey", "password", "token", "secret"]):
                redacted[k] = "***"
            else:
                redacted[k] = _redact(v)
        return redacted
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


class _CallRecord:
    def __init__(self, tag: str, func_name: str, args: tuple, kwargs: dict) -> None:
        self.tag = tag
        self.func_name = func_name
        self.args = args
        self.kwargs = kwargs
        self.notes: List[Dict[str, Any]] = []
        self.start_ts = time.time()
        self.end_ts: Optional[float] = None
        self.return_value: Any = None
        self.exception: Optional[BaseException] = None


_stack_var: contextvars.ContextVar[List[_CallRecord]] = contextvars.ContextVar("_debug_stack", default=[])


class DebugInfoManager:
    """
    轻量调试信息管理器：
    - 通过装饰器自动记录入参与出参（函数提供方无需在调用处决定打印内容）
    - 函数内部可调用 note() 注册关键中间变量
    - 在函数结束时统一打印本次调用的调试信息
    """

    def start(self, tag: str, func_name: str, args: tuple, kwargs: dict) -> None:
        stack = list(_stack_var.get())
        stack.append(_CallRecord(tag, func_name, args, kwargs))
        _stack_var.set(stack)

    def note(self, key: str, value: Any) -> None:
        stack = _stack_var.get()
        if not stack:
            return
        stack[-1].notes.append({"key": key, "value": value})

    def end(self, return_value: Any = None, exception: Optional[BaseException] = None) -> None:
        stack = list(_stack_var.get())
        if not stack:
            return
        rec = stack.pop()
        _stack_var.set(stack)

        rec.end_ts = time.time()
        rec.return_value = return_value
        rec.exception = exception

        # 打印
        duration_ms = int((rec.end_ts - rec.start_ts) * 1000) if rec.end_ts else -1
        print(f"=== 调试 · {rec.tag} · {rec.func_name} · {duration_ms}ms ===")

        safe_args = _redact(rec.args)
        safe_kwargs = _redact(rec.kwargs)
        print(f"入参(args): {_safe_serialize(safe_args)}")
        print(f"入参(kwargs): {_safe_serialize(safe_kwargs)}")

        if rec.notes:
            print("中间记录(notes):")
            for n in rec.notes:
                print(f"  - {n.get('key')}: {_safe_serialize(_redact(n.get('value')))}")

        if rec.exception is not None:
            print(f"异常: {type(rec.exception).__name__}: {rec.exception}")
        else:
            print(f"返回: {_safe_serialize(_redact(rec.return_value))}")
        print("-" * 60)


def dispInfo(tag: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    调试信息装饰器：
    - 自动记录入参与返回值
    - 支持函数内部通过 debug.note(key, value) 记录中间状态
    - 函数结束时统一打印完整调试信息
    """

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def _sync_wrapper(*args, **kwargs):
            debug.start(tag, func.__name__, args, kwargs)
            try:
                rv = func(*args, **kwargs)
                debug.end(rv, None)
                return rv
            except BaseException as e:
                debug.end(None, e)
                raise

        return _sync_wrapper

    return _decorator


# 全局实例
debug = DebugInfoManager()


