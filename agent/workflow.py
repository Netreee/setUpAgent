from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from agent.task_types import AgentState
from agent.planner import plan_with_llm
from agent.tools import (
    RUN_INSTRUCTION_TOOL,
    make_tool_call_message,
    extract_last_tool_result,
)
from agent.observer import observe
from agent.executor import decide_next_action
from agent.debug import dispInfo, debug


@dispInfo("workflow")
def plan_node(state: AgentState) -> AgentState:
    """
    规划节点：使用 LLM 生成线性步骤计划。
    输入：state 中应包含 messages[{-1}.content] 或者 state['task']['goal']
    输出：写入 task, plan_text, current_step_index
    """
    goal = state.get("task", {}).get("goal")
    if not goal:
        # 兼容从 messages 读取
        messages = state.get("messages", [])
        goal = messages[-1]["content"] if messages else ""

    task, plan_text, raw_plan = plan_with_llm(goal)
    debug.note("plan_raw", raw_plan)
    debug.note("summary_plan", plan_text)
    return {
        **state,
        "task": task,
        "plan_text": plan_text,
        "plan_raw": raw_plan,
        "current_step_index": 0,
        "last_result": None,
        "is_complete": False,
        "failed": False,
        "observation": "已生成计划",
    }


@dispInfo("workflow")
def decide_node(state: AgentState) -> AgentState:
    """
    决策节点：基于计划与上一步执行结果，判断下一步操作：
    - 产生工具调用（写入 messages 中的 tool_calls，由 ToolNode 执行）
    - 或请求重规划（设置 replan_requested/route）
    """
    task = state.get("task", {})
    steps = task.get("steps", [])
    idx = int(state.get("current_step_index", 0))
    last_result = state.get("last_result")

    decision = decide_next_action(task, idx, last_result)
    if decision.get("action") == "replan":
        # 请求重规划
        debug.note("route", "plan")
        return {
            **state,
            "replan_requested": True,
            "route": "plan",
            "decide_raw": decision.get("raw", ""),
        }

    # 默认或明确 call_tool：生成工具调用消息（使用持久会话）
    nl_instruction = str(decision.get("nl_instruction", "")).strip()
    timeout = int(decision.get("timeout", 60) or 60)
    # 决策可直接给出 token，否则沿用 state 上保存的
    session_token = decision.get("session_token") or state.get("session_id")

    messages = list(state.get("messages", []))
    messages.append(make_tool_call_message(nl_instruction, timeout, session_token))
    debug.note("tool_in_nl_instruction", nl_instruction)
    debug.note("tool_in_timeout", timeout)
    debug.note("route", "execute")
    return {**state, "messages": messages, "route": "execute", "decide_raw": decision.get("raw", "")}


@dispInfo("workflow")
def observe_node(state: AgentState) -> AgentState:
    """
    观察节点：根据 last_result 和 current_step_index 判断是否完成或失败。
    """
    task = state["task"]
    idx = int(state.get("current_step_index", 0))
    # 从消息中提取最近一次工具结果
    tool_result = extract_last_tool_result(state.get("messages", []))
    last_result = tool_result or state.get("last_result")
    if last_result:
        debug.note("tool_out_command", last_result.get("command"))
        debug.note("tool_out_exit_code", last_result.get("exit_code"))
        # 执行阶段摘要（单行）
        try:
            cmd = str(last_result.get("command", ""))
            code = last_result.get("exit_code", "")
            debug.note("summary_execute", f"命令: {cmd} | 退出码: {code}")
        except Exception:
            pass

    # 使用观察器评估
    # 注入 step_id 以便观察器有更稳健的上下文
    if last_result is not None and isinstance(last_result, dict):
        try:
            current_step = task.get("steps", [])[idx] if 0 <= idx < len(task.get("steps", [])) else None
            if current_step and "step_id" not in last_result:
                last_result = {**last_result, "step_id": current_step.get("id")}
        except Exception:
            pass

    is_complete, failed, obs = observe(task, idx, last_result)
    debug.note("observation", obs)

    # 成功推进索引（当有最新工具结果且成功时）
    next_idx = idx
    if tool_result and int(tool_result.get("exit_code", -1)) == 0:
        next_idx = idx + 1

    # 若成功推进且 next_idx 已达末尾，标记完成并终止
    steps_len = len(task.get("steps", []))
    completed_now = (not failed) and (next_idx >= steps_len)

    # 如果工具返回了新的 session_token（会话被新建或重建），更新到状态
    new_token = None
    try:
        if isinstance(tool_result, dict):
            new_token = tool_result.get("session_token")
    except Exception:
        pass

    result_state = {
        **state,
        "current_step_index": next_idx,
        "last_result": last_result,
        "is_complete": True if completed_now else is_complete,
        "failed": False if completed_now else failed,
        "observation": obs,
        # 失败时请求重规划
        "replan_requested": False if completed_now else bool(failed),
        "route": "plan" if (failed and not completed_now) else (END if completed_now else "execute"),
        "session_id": new_token or state.get("session_id"),
    }
    debug.note("route", result_state.get("route"))
    # 观察阶段摘要（单行）
    try:
        route = result_state.get("route")
        debug.note("summary_observe", f"观察: {obs} | 路由: {route}")
    except Exception:
        pass
    return result_state


def create_task_graph():
    """构建任务执行 LangGraph 工作流（plan → execute → observe → (END|plan)）。"""
    workflow = StateGraph(AgentState)
    workflow.add_node("plan", plan_node)
    # 决策节点：决定下一步调用的工具或是否重规划
    workflow.add_node("decide", decide_node)
    # ToolNode: 将 run_single_sync 暴露为工具
    workflow.add_node("execute", ToolNode([RUN_INSTRUCTION_TOOL]))
    workflow.add_node("observe", observe_node)

    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "decide")
    workflow.add_edge("decide", "execute")
    workflow.add_edge("execute", "observe")
    # 根据观察路由决定流向：END / plan(重规划) / execute(继续)
    def _route(s: AgentState):
        if s.get("is_complete"):
            return END
        route = s.get("route")
        if route == "plan" or s.get("replan_requested"):
            return "plan"
        # 默认回到决策节点继续推进
        return "decide"

    workflow.add_conditional_edges(
        "observe",
        _route,
        {END: END, "plan": "plan", "decide": "decide"},
    )

    return workflow.compile()


