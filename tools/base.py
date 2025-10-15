"""工具基础设施：统一返回格式"""

import json
from typing import Any, Dict, Optional


def tool_response(
    *,
    tool: str,
    ok: bool,
    data: Dict[str, Any],
    error: Optional[str] = None,
) -> str:
    """构造统一的工具返回格式
    
    Args:
        tool: 工具名称（如 "files_read", "run_instruction"）
        ok: 工具执行是否成功
        data: 工具特定的结构化数据
        error: 错误信息（仅失败时）
    
    Returns:
        JSON字符串
    """
    result = {
        "ok": ok,
        "tool": tool,
        "data": data,
    }
    if error is not None:
        result["error"] = error
    return json.dumps(result, ensure_ascii=False)


def parse_tool_response(json_str: str) -> Dict[str, Any]:
    """解析工具返回
    
    Returns:
        {
            "ok": bool,
            "tool": str,
            "data": dict,
            "error": str | None
        }
    """
    try:
        result = json.loads(json_str)
        return {
            "ok": result.get("ok", False),
            "tool": result.get("tool", "unknown"),
            "data": result.get("data", {}),
            "error": result.get("error"),
        }
    except Exception:
        return {
            "ok": False,
            "tool": "unknown",
            "data": {},
            "error": "invalid_json"
        }

