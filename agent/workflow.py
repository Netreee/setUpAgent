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
from agent.observer import observe, observe_v2
import os
from agent.executor import decide_next_action
from config import get_config
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

    # 将 mode/episode/facts/finished_titles 作为上下文传递给 planner（增量重规划时也用）
    context = {
        "mode": state.get("mode", "discover"),
        "episode": int(state.get("episode", 1) or 1),
        "facts": dict(state.get("facts", {})),
        "finished_titles": list(state.get("finished_titles", [])),
    }
    # 如果已有任务，补充增量上下文
    prev_task = state.get("task")
    if isinstance(prev_task, dict):
        steps = prev_task.get("steps", [])
        idx = int(state.get("current_step_index", 0) or 0)
        context.update({
            "completed_steps": steps[: max(0, min(idx, len(steps)))],
            "remaining_steps": steps[max(0, min(idx, len(steps))):],
            "last_result": state.get("last_result", {}),
        })

    task, plan_text, raw_plan = plan_with_llm(goal, context=context)
    debug.note("plan_raw", raw_plan)
    debug.note("summary_plan", plan_text)
    return {
        **state,
        "task": task,
        "plan_text": plan_text,
        "plan_raw": raw_plan,
        # 保持当前索引，避免回到0
        "current_step_index": int(state.get("current_step_index", 0) or 0),
        "last_result": None,
        "is_complete": False,
        "failed": False,
        "observation": "已生成计划",
        # 保留现有 mode/episode/facts，不重置
        "mode": state.get("mode", "discover"),
        "episode": int(state.get("episode", 1) or 1),
        "facts": dict(state.get("facts", {})),
        "READMEinfo": dict(state.get("READMEinfo", {})),
        "finished_titles": list(state.get("finished_titles", [])),
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

    decision = decide_next_action(
        task,
        idx,
        last_result,
        mode=str(state.get("mode", "")),
        episode=int(state.get("episode", 0) or 0),
        facts=state.get("facts", {}),
    )
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

    # 在调用工具前，将 REPO_ROOT/PROJECT_ROOT 注入到当前进程环境，供执行层使用
    try:
        facts_env = state.get("facts", {}) or {}
        repo_root = facts_env.get("repo_root") or facts_env.get("repo_path") or ""
        if isinstance(repo_root, str) and repo_root:
            os.environ["REPO_ROOT"] = repo_root
        project_root = facts_env.get("project_root") or ""
        if isinstance(project_root, str) and project_root:
            os.environ["PROJECT_ROOT"] = project_root
    except Exception:
        pass

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

    # 新的 LLM 路由与事实增量
    route_decision = observe_v2(
        task,
        idx,
        last_result,
        mode=str(state.get("mode", "discover")),
        episode=int(state.get("episode", 1) or 1),
        facts=state.get("facts", {}),
    )
    debug.note("observation_decision", route_decision)
    obs = str(route_decision.get("notes", "")).strip() or ""

    # 合并 facts（并规范字段名）
    facts = dict(state.get("facts", {}))
    try:
        delta = route_decision.get("facts_delta") or {}
        if isinstance(delta, dict):
            facts.update(delta)
    except Exception:
        pass
    # 统一仓库路径字段
    if "repo_path" in facts and not facts.get("repo_root"):
        facts["repo_root"] = facts.pop("repo_path")
    if facts.get("repo_root") and not facts.get("exec_root"):
        facts["exec_root"] = facts["repo_root"]
    # 移除不使用的旧字段
    facts.pop("work_dir", None)

    # 若本次为成功的 git clone，自动推导并回填 project_root/project_name，并保持 repo_root 指向工作区根
    try:
        if isinstance(last_result, dict) and int(last_result.get("exit_code", -1)) == 0:
            cmd_text = str(last_result.get("command", ""))
            if cmd_text and "git clone" in cmd_text.lower():
                import re
                # 提取 URL 与可选目标路径（可能为 Join-Path 表达式或裸路径）
                m = re.search(r"\bgit\s+clone\s+(\S+)(?:\s+(\([^\)]*\)|\"[^\"]+\"|'[^']+'|[^\s]+))?", cmd_text, re.IGNORECASE)
                if m:
                    url = (m.group(1) or "").strip().strip("'\"")
                    target_arg = (m.group(2) or "").strip()

                    # 从 URL 推导仓库名
                    repo_name = url.rstrip("/")
                    if "/" in repo_name:
                        repo_name = repo_name.split("/")[-1]
                    if "\\" in repo_name:
                        repo_name = repo_name.split("\\")[-1]
                    if repo_name.lower().endswith(".git"):
                        repo_name = repo_name[:-4]
                    repo_name = repo_name or "repo"

                    # 计算工作区根：优先环境 REPO_ROOT，其次配置的 agent_work_root，最后 CWD
                    try:
                        workspace_root = os.environ.get("REPO_ROOT") or get_config().agent_work_root or os.getcwd()
                    except Exception:
                        workspace_root = os.environ.get("REPO_ROOT") or os.getcwd()

                    project_root_abs: str = ""
                    # 解析 Join-Path $env:REPO_ROOT 'xxx'
                    if target_arg:
                        m_join_s = re.search(r"Join-Path\s+\$env:REPO_ROOT\s+'([^']+)'", target_arg, re.IGNORECASE)
                        m_join_d = re.search(r'Join-Path\s+\$env:REPO_ROOT\s+"([^"]+)"', target_arg, re.IGNORECASE)
                        repo_rel = None
                        if m_join_s:
                            repo_rel = m_join_s.group(1)
                        elif m_join_d:
                            repo_rel = m_join_d.group(1)
                        if repo_rel:
                            project_root_abs = os.path.normpath(os.path.join(workspace_root, repo_rel))
                        else:
                            # 其他形式：去掉括号与引号后按相对/绝对路径处理
                            targ_clean = target_arg.strip().strip("()").strip().strip("'\"")
                            if targ_clean:
                                if os.path.isabs(targ_clean):
                                    project_root_abs = os.path.normpath(targ_clean)
                                else:
                                    project_root_abs = os.path.normpath(os.path.join(workspace_root, targ_clean))

                    if not project_root_abs:
                        project_root_abs = os.path.normpath(os.path.join(workspace_root, repo_name))

                    # 写回事实：保持 repo_root = 工作区根；project_root = 克隆目录（覆盖占位符）
                    if not facts.get("repo_root"):
                        facts["repo_root"] = workspace_root
                    # project_root 若为占位符（如 repo_root/... 或 $env:REPO_ROOT/...）或未设置，则使用绝对路径覆盖
                    prv = str(facts.get("project_root", ""))
                    if (not prv) or prv.lower().startswith("repo_root/") or prv.lower().startswith("repo_root\\") or prv.startswith("$env:REPO_ROOT"):
                        facts["project_root"] = project_root_abs
                    if not facts.get("project_name"):
                        facts["project_name"] = repo_name
                    if not facts.get("exec_root") and facts.get("repo_root"):
                        # 执行根保持为工作区根
                        facts["exec_root"] = facts["repo_root"]
    except Exception:
        pass

    # 若上一步为读取 README 的输出，则尝试解析并合并到 READMEinfo
    readmeinfo = dict(state.get("READMEinfo", {}))
    try:
        from agent.observer import extract_readme_info
        if isinstance(last_result, dict):
            cmd = str(last_result.get("command", ""))
            out_text = str(last_result.get("stdout", ""))
            if out_text and (
                "README" in cmd.upper() or "README" in out_text[:50].upper()
            ):
                parsed = extract_readme_info(out_text)
                if isinstance(parsed, dict) and parsed:
                    readmeinfo.update(parsed)
    except Exception:
        pass

    # 若本次工具执行成功，则把当前步骤标题添加到 finished_titles（避免重规划重复）
    finished_titles = list(state.get("finished_titles", []))
    try:
        if tool_result and int(tool_result.get("exit_code", -1)) == 0:
            if 0 <= idx < len(task.get("steps", [])):
                title = task.get("steps", [])[idx].get("title")
                if isinstance(title, str) and title and title not in finished_titles:
                    finished_titles.append(title)
    except Exception:
        pass

    # 索引推进策略（以 route 为准）
    next_idx = idx
    route = route_decision.get("route", "decide")
    if route == "repeat_step":
        next_idx = idx
    elif route == "skip_step":
        next_idx = idx + 1
    elif route == "decide":
        # 若工具成功且未要求 repeat/skip，则推进
        if tool_result and int(tool_result.get("exit_code", -1)) == 0:
            next_idx = idx + 1
        else:
            next_idx = idx
    elif route == "plan":
        # 交由 plan 重新规划，索引按需要可保持
        next_idx = idx
    elif route == "end":
        next_idx = idx

    # 如果工具返回了新的 session_token（会话被新建或重建），更新到状态
    new_token = None
    try:
        if isinstance(tool_result, dict):
            new_token = tool_result.get("session_token")
    except Exception:
        pass

    # 结束判定以 route 决定
    steps_len = len(task.get("steps", []))
    is_complete = True if route == "end" else False
    failed = False
    # 模式切换：observer 可在任何路由返回中附带 mode 建议
    new_mode = state.get("mode", "discover")
    new_episode = int(state.get("episode", 1) or 1)
    replan_requested = False
    next_route = "decide"
    if route == "plan":
        replan_requested = True
        next_route = "plan"
    elif route == "decide":
        next_route = "decide"
    elif route == "repeat_step":
        next_route = "decide"
    elif route == "skip_step":
        next_route = "decide"
    elif route == "end":
        next_route = END

    # 应用任意路由下的 mode 更新建议（若有变化则自增 episode 并触发重规划，除非已结束）
    try:
        suggested_mode = route_decision.get("mode")
        if suggested_mode in ("discover", "execute") and suggested_mode != new_mode:
            new_mode = suggested_mode
            new_episode = new_episode + 1
            if next_route != END:
                replan_requested = True
                next_route = "plan"
    except Exception:
        pass

    result_state = {
        **state,
        "current_step_index": next_idx,
        "last_result": last_result,
        "is_complete": is_complete,
        "failed": failed,
        "observation": obs,
        "replan_requested": replan_requested,
        "route": next_route,
        "session_id": new_token or state.get("session_id"),
        "mode": new_mode,
        "episode": new_episode,
        "facts": facts,
        "READMEinfo": readmeinfo,
    }
    debug.note("route", result_state.get("route"))
    # 观察阶段摘要（单行）
    try:
        route = result_state.get("route")
        debug.note("summary_observe", f"观察: {obs} | 路由: {route}")
    except Exception:
        pass
    # 将本次 observer 结束后的完整状态写入调试日志
    try:
        debug.write_json_log(result_state)
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
    # 根据观察路由决定流向：END / plan(重规划) / decide(继续)
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


