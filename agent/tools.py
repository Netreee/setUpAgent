from typing import Any, Dict, List
import json

from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage

from run_singleV2 import run_single as run_single_v2
from agent.debug import dispInfo, debug
from agent.async_utils import run_coro_sync
import json


@tool("run_instruction")
# @dispInfo("tool")
def RUN_INSTRUCTION_TOOL(nl_instruction: str, timeout: int = 60, session_token: str | None = None) -> str:
    """
    将自然语言指令转换为单条可执行命令并在持久 PowerShell 会话中执行（V2）。
    返回JSON字符串：{"exit_code": int, "stdout": str, "stderr": str, "command": str, "work_dir": str, "session_token": str}
    """
    debug.note("nl_instruction", nl_instruction)
    debug.note("timeout", timeout)
    debug.note("session_token_in", session_token or "<none>")

    try:
        token, result = run_coro_sync(
            run_single_v2(nl_instruction, timeout=timeout, session_token=session_token),
            timeout=timeout + 5,
        )
        # 附带会话 token 以便上层保存
        result = {**result, "session_token": token}
        debug.note("session_token_out", token)
        debug.note("run_single_result", result)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        # 工具内部异常时也返回结构化JSON，便于上层观察与重规划
        err_result: Dict[str, Any] = {
            "exit_code": 1,
            "stdout": "",
            "stderr": f"{type(e).__name__}: {e}",
            "command": "",
            "work_dir": "",
            "session_token": session_token,
        }
        debug.note("run_single_error", str(e))
        return json.dumps(err_result, ensure_ascii=False)


def make_tool_call_message(nl_instruction: str, timeout: int = 60, session_token: str | None = None) -> AIMessage:
    """构造一个请求执行工具的AI消息，用于驱动ToolNode。支持会话token。"""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "run_instruction",
                "args": {"nl_instruction": nl_instruction, "timeout": timeout, "session_token": session_token},
                "id": "call_run_instruction",
            }
        ],
    )


def extract_last_tool_result(messages: List[Any]) -> Dict[str, Any]:
    """
    从消息列表中提取最近一次工具执行结果。
    期望由 ToolNode 追加的 ToolMessage，内容为JSON字符串。
    """
    for msg in reversed(messages or []):
        if isinstance(msg, ToolMessage):
            try:
                return json.loads(msg.content)
            except Exception:
                # 若内容不是合法JSON，则视为工具错误，包装为失败结果以触发重规划
                text = ""
                try:
                    text = str(msg.content)
                except Exception:
                    text = ""
                return {
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": text or "Tool execution error",
                    "command": "",
                    "work_dir": "",
                }
    return {}


def _build_windows_prompt(nl_instruction: str, work_dir: str) -> str:
    work_str = str(work_dir).replace('\\', '\\\\')
    return f"""
You are a helpful assistant that translates natural language into a single, self-contained shell command.
Rules:
- Output ONLY the command, no explanation, no quotes.
- Assume cwd is {work_str}
- IMPORTANT: Windows system. Use PowerShell commands, avoid Unix tools.
- Use single backslashes in Windows paths; quote only when path contains spaces.
- Prefer simple, non-interactive commands.

User request: {nl_instruction}
"""


