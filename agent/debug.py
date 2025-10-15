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

        # 构建日志内容
        duration_ms = int((rec.end_ts - rec.start_ts) * 1000) if rec.end_ts else -1
        lines = [f"=== 调试 · {rec.tag} · {rec.func_name} · {duration_ms}ms ==="]

        # 只输出kwargs（args通常为空或不重要）
        safe_kwargs = _redact(rec.kwargs)
        if safe_kwargs:
            lines.append(f"入参: {_safe_serialize(safe_kwargs, limit=400)}")

        if rec.notes:
            lines.append("关键信息:")
            for n in rec.notes:
                key = n.get('key', '')
                # 跳过prompt（太长且无用）
                if 'prompt' in key.lower():
                    continue
                val = n.get('value')
                # 对raw_resp等响应类信息，限制在200字符
                limit = 200 if 'resp' in key.lower() else 300
                lines.append(f"  • {key}: {_safe_serialize(_redact(val), limit=limit)}")

        if rec.exception is not None:
            lines.append(f"❌ 异常: {type(rec.exception).__name__}: {rec.exception}")
        else:
            # 对返回值进行精简，特别是JSON字符串
            ret_str = _safe_serialize(_redact(rec.return_value), limit=500)
            lines.append(f"✓ 返回: {ret_str}")
        lines.append("-" * 60)
        
        # 写入文件
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            with open(".agent_debug.log", "a", encoding="utf-8") as f:
                f.write(f"\n[{ts}]\n")
                f.write("\n".join(lines))
                f.write("\n")
        except Exception:
            # 如果文件写入失败，回退到控制台输出
            for line in lines:
                print(line)

    # 追加：将任意对象（如完整 state）以 JSON 形式写入日志文件
    def write_json_log(self, payload: Any, file_path: Optional[str] = None) -> None:
        try:
            # 默认日志文件：.agent_state.log 放在当前工作目录
            target = file_path or ".agent_state.log"
            # 尽量保留原始对象，但要做脱敏与可序列化处理
            redacted = _redact(payload)
            text = json.dumps(redacted, ensure_ascii=False, default=str)
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            with open(target, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] ")
                f.write(text)
                f.write("\n")
        except Exception:
            # 不因日志失败影响主流程
            pass


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

        # 兼容 async 函数：如果 func 是协程函数，则返回异步包装器
        try:
            import inspect
            if inspect.iscoroutinefunction(func):
                @wraps(func)
                async def _async_wrapper(*args, **kwargs):
                    debug.start(tag, func.__name__, args, kwargs)
                    try:
                        rv = await func(*args, **kwargs)
                        debug.end(rv, None)
                        return rv
                    except BaseException as e:
                        debug.end(None, e)
                        raise
                return _async_wrapper  # type: ignore[return-value]
        except Exception:
            pass

        return _sync_wrapper

    return _decorator


# 全局实例
debug = DebugInfoManager()


