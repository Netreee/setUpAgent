from typing import Optional, Dict, Any
from agent.task_types import Task
from utils import llm_completion
from agent.debug import dispInfo, debug


def _short(text: Optional[str], n: int = 600) -> str:
    return (text or "")[:n]


# @dispInfo("decider")
def summarize_context(task: Task, current_index: int, last_result: Optional[Dict[str, Any]]) -> str:
    """汇总上下文（目标/计划/当前位置/上一次结果），用于喂给LLM。"""
    steps = task.get("steps", [])
    plan_titles = [s.get("title", f"步骤{i+1}") for i, s in enumerate(steps)]
    current_step = steps[current_index] if 0 <= current_index < len(steps) else None

    return (
        f"任务目标: {task.get('goal','')}\n"
        f"计划步骤（标题序列）: {plan_titles}\n"
        f"当前步骤索引: {current_index}\n"
        f"当前步骤详情: {current_step}\n"
        f"上一次结果: exit_code={last_result.get('exit_code') if last_result else None}, "
        f"command={last_result.get('command') if last_result else None}\n"
        f"STDOUT: {_short(last_result.get('stdout') if last_result else '')}\n"
        f"STDERR: {_short(last_result.get('stderr') if last_result else '')}\n"
    )


# @dispInfo("decider")
def decide_next_action(task: Task, current_index: int, last_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    决策下一步操作：
    返回结构：{"action": "call_tool"|"replan", "nl_instruction": str?, "timeout": int?, "session_token": str?}
    """
    context = summarize_context(task, current_index, last_result)

    prompt = (
        "你是一个任务执行的指挥官，负责根据当前计划与最近一次执行结果，决定下一步操作。\n"
        "环境：Windows + PowerShell；工作目录由系统管理；你只需要输出结构化决策。\n\n"
        f"{context}\n"
        "输出严格的JSON，不要多余文字：\n"
        '{"action": "call_tool"|"replan", "nl_instruction": "...", "timeout": 60, "session_token": "..."}。当需要重规划时可省略 nl_instruction。\n'
        "严格要求：nl_instruction 必须是自然语言描述（做什么），不要直接给出命令行。命令行由系统自动生成。\n"
        "当上一步失败且可以通过小改动修复时，优先 call_tool 并给出更稳健的自然语言指令；确需大改动再标记 replan。"
    )

    debug.note("decide_prompt", prompt)
    try:
        resp = llm_completion(prompt, temperature=0.2, max_tokens=300).strip()
    except Exception:
        resp = "{}"
    debug.note("decide_raw_resp", resp)

    import json, re
    data: Optional[Dict[str, Any]] = None
    try:
        data = json.loads(resp)
    except Exception:
        m = re.search(r'\{.*\}', resp, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None

    if not isinstance(data, dict):
        # 回退：无有效决策时，尝试继续按计划执行
        steps = task.get("steps", [])
        step = steps[current_index] if 0 <= current_index < len(steps) else None
        if not step:
            return {"action": "replan", "raw": resp}
        return {
            "action": "call_tool",
            "nl_instruction": str(step.get("instruction", "")).strip(),
            "timeout": int(step.get("timeout") or 60),
            "raw": resp,
        }

    action = data.get("action")
    if action == "replan":
        return {"action": "replan", "raw": resp}

    # 默认/明确 call_tool
    nl_instruction = str(data.get("nl_instruction", "")).strip()
    timeout = int(data.get("timeout", 60) or 60)
    session_token = data.get("session_token")
    if not nl_instruction:
        # 回退计划步
        steps = task.get("steps", [])
        step = steps[current_index] if 0 <= current_index < len(steps) else None
        if not step:
            return {"action": "replan", "raw": resp}
        nl_instruction = str(step.get("instruction", "")).strip()
        timeout = int(step.get("timeout") or 60)
        # 默认沿用 last_result 中的 token（如果存在）
        if isinstance(last_result, dict):
            session_token = last_result.get("session_token") or session_token

    return {"action": "call_tool", "nl_instruction": nl_instruction, "timeout": timeout, "session_token": session_token, "raw": resp}
