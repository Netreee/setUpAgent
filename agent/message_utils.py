from typing import Any, Dict, List
import json

from langchain_core.messages import AIMessage, ToolMessage


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


def make_generic_tool_call_message(tool_name: str, tool_args: Dict[str, Any] | None = None) -> AIMessage:
    """构造一个通用的工具调用AI消息，支持任意已注册工具名称与参数。"""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": tool_name,
                "args": dict(tool_args or {}),
                "id": f"call_{tool_name}",
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


