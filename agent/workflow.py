from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from agent.task_types import AgentState
from agent.planner import plan_with_llm
from agent.message_utils import (
    make_tool_call_message,
    make_generic_tool_call_message,
    extract_last_tool_result,
)
from tools import (
    RUN_INSTRUCTION_TOOL,
    FILES_EXISTS_TOOL,
    FILES_STAT_TOOL,
    FILES_LIST_TOOL,
    FILES_READ_TOOL,
    FILES_FIND_TOOL,
    PYENV_PYTHON_INFO_TOOL,
    PYENV_TOOL_VERSIONS_TOOL,
    PYENV_PARSE_PYPROJECT_TOOL,
    PYENV_SELECT_INSTALLER_TOOL,
    GIT_REPO_STATUS_TOOL,
    GIT_ENSURE_CLONED_TOOL,
)
from agent.observer import observe, observe_v2
from agent.discover_react import run_discover_react
import os
import re
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
    curr_mode = state.get("mode", "discover")
    context = {
        "mode": curr_mode,
        "episode": int(state.get("episode", 1) or 1),
        "facts": dict(state.get("facts", {})),
        "finished_titles": list(state.get("finished_titles", [])),
    }
    
    # 检测模式切换：若从上一个任务的隐式模式切换到当前模式，需要完全重规划
    prev_task = state.get("task")
    mode_switched = False
    if isinstance(prev_task, dict):
        # 从 finished_titles 推断上一个任务的模式（简单启发式）
        prev_mode = state.get("_last_plan_mode", "discover")  # 保存上次规划时的模式
        if prev_mode != curr_mode:
            mode_switched = True
            debug.note("mode_switch_detected", f"{prev_mode} -> {curr_mode}")
    
    # 如果已有任务，补充增量上下文
    if isinstance(prev_task, dict) and not mode_switched:
        # 增量重规划：保留已完成步骤
        steps = prev_task.get("steps", [])
        idx = int(state.get("current_step_index", 0) or 0)
        context.update({
            "completed_steps": steps[: max(0, min(idx, len(steps)))],
            "remaining_steps": steps[max(0, min(idx, len(steps))):],
            "last_result": state.get("last_result", {}),
        })
    else:
        # 模式切换或首次规划：完全重规划（不保留已完成步骤）
        context.update({
            "completed_steps": [],
            "remaining_steps": [],
            "last_result": state.get("last_result", {}),
        })

    task, plan_text, raw_plan = plan_with_llm(goal, context=context)
    debug.note("plan_raw", raw_plan)
    debug.note("summary_plan", plan_text)
    
    # 如果是模式切换后的完全重规划，重置步骤索引到 0
    next_idx = 0 if mode_switched else int(state.get("current_step_index", 0) or 0)
    
    return {
        **state,
        "task": task,
        "plan_text": plan_text,
        "plan_raw": raw_plan,
        # 保持当前索引，避免回到0（除非模式切换）
        "current_step_index": next_idx,
        "last_result": None,
        "is_complete": False,
        "failed": False,
        "observation": "已生成计划" + (" (模式切换)" if mode_switched else ""),
        # 保留现有 mode/episode/facts，不重置
        "mode": curr_mode,
        "episode": int(state.get("episode", 1) or 1),
        "facts": dict(state.get("facts", {})),
        "READMEinfo": dict(state.get("READMEinfo", {})),
        # 清理重规划标志，避免无限循环
        "replan_requested": False,
        "route": None,
        "finished_titles": list(state.get("finished_titles", [])),
        # 保存本次规划时的模式，用于下次检测模式切换
        "_last_plan_mode": curr_mode,
    }


@dispInfo("workflow")
def discover_node(state: AgentState) -> AgentState:
    """
    Discover entry node: run the ReAct discover agent once to collect facts and a summary,
    then hand control to planner/decider pipeline.
    """
    goal = None
    try:
        goal = state.get("task", {}).get("goal")
    except Exception:
        goal = None
    if not goal:
        messages = state.get("messages", [])
        goal = messages[-1]["content"] if messages else ""

    seed_facts = {}
    try:
        seed_facts = dict(state.get("facts", {}) or {})
    except Exception:
        seed_facts = {}

    try:
        out = run_discover_react(goal, seed_facts=seed_facts)
    except Exception as e:
        out = {"facts": seed_facts, "summary": f"(discover failed) {type(e).__name__}: {e}", "transcript": []}

    facts = dict(out.get("facts", {}) or {})
    discover_summary = str(out.get("summary", ""))
    discover_transcript = list(out.get("transcript", []) or [])

    # Normalize minimal expected keys
    try:
        if "repo_path" in facts and not facts.get("repo_root"):
            facts["repo_root"] = facts.pop("repo_path")
        if facts.get("repo_root") and not facts.get("exec_root"):
            facts["exec_root"] = facts["repo_root"]
        facts.pop("work_dir", None)
    except Exception:
        pass

    return {
        **state,
        "facts": facts,
        "discover_summary": discover_summary,
        "discover_transcript": discover_transcript,
        # Set default mode to execute after discover; subsequent planning will rely on facts
        "mode": "execute",
        "episode": int(state.get("episode", 1) or 1),
        "route": "plan",
        "replan_requested": True,
        "observation": "已完成 ReAct 发现阶段并写入 facts",
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

    # 根据决策生成调用消息（支持 call_tool/call_instruction）
    action = decision.get("action")
    timeout = int(decision.get("timeout", 60) or 60)
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

    # 反克隆保护与替换：将任何即将执行的 "git clone" 转换为结构化 git 工具
    try:
        facts_env = state.get("facts", {}) or {}
        planned_nl = str(decision.get("nl_instruction", "") or "")
        is_clone_instruction = isinstance(planned_nl, str) and re.search(r"\bgit\s+clone\b", planned_nl, re.IGNORECASE)
        if action == "call_instruction" and is_clone_instruction:
            # 若已存在 project_root，则改为查询仓库状态，避免重复克隆
            proj = facts_env.get("project_root") or ""
            if isinstance(proj, str) and proj:
                try:
                    if os.path.exists(proj):
                        messages.append(make_generic_tool_call_message("git_repo_status", {"path": proj}))
                        debug.note("clone_guard", f"project_root exists, redirect to git_repo_status: {proj}")
                        debug.note("route", "execute")
                        return {**state, "messages": messages, "decide_raw": decision.get("raw", "")}
                except Exception:
                    pass
            # 未存在则尝试解析 URL/目标目录，调用 git_ensure_cloned
            m = re.search(r"\bgit\s+clone\s+(\S+)(?:\s+(\S+))?", planned_nl, re.IGNORECASE)
            url = m.group(1) if m else ""
            dest_arg = m.group(2) if m else None
            args = {"url": url}
            if isinstance(dest_arg, str) and dest_arg and not dest_arg.startswith("$"):
                # 清理不合法的目标：dest='.' 等同未提供，避免将工作区根当作仓库目录
                clean_dest = dest_arg.strip().strip('"').strip("'")
                if clean_dest not in ("", ".", "./"):
                    args["dest"] = clean_dest
            elif isinstance(proj, str) and proj:
                # 使用事实中的 project_root 作为目标
                args["dest"] = proj
            args["depth"] = 1
            args["sparse"] = True
            messages.append(make_generic_tool_call_message("git_ensure_cloned", args))
            debug.note("clone_rewrite", {"from": planned_nl, "to": args})
            debug.note("route", "execute")
            return {**state, "messages": messages, "decide_raw": decision.get("raw", "")}
    except Exception:
        pass

    if action == "call_tool":
        tool_name = str(decision.get("tool_name", "")).strip()
        tool_args = decision.get("tool_args") or {}
        messages.append(make_generic_tool_call_message(tool_name, tool_args))
        debug.note("tool_in_name", tool_name)
        debug.note("tool_in_args", tool_args)
    else:
        nl_instruction = str(decision.get("nl_instruction", "")).strip()
        messages.append(make_tool_call_message(nl_instruction, timeout, session_token))
        debug.note("tool_in_nl_instruction", nl_instruction)
        debug.note("tool_in_timeout", timeout)
    debug.note("route", "execute")
    return {**state, "messages": messages, "decide_raw": decision.get("raw", "")}


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
    observer_route = route_decision.get("route", "decide")  # 保存原始路由
    
    if observer_route == "repeat_step":
        next_idx = idx
    elif observer_route == "skip_step":
        next_idx = idx + 1
    elif observer_route == "decide":
        # 若观察器未要求 repeat/skip，则依据观察器显式 success 或工具类 ok(true) 决定是否推进
        is_success = False
        try:
            # 优先采用 LLM 观察结果中的 success 信号
            if route_decision.get("success") is True:
                is_success = True
            elif route_decision.get("success") is False:
                is_success = False
            else:
                # 兜底：仅对非 shell 的只读工具（如 files/pyenv）使用 ok=true 判定
                if isinstance(tool_result, dict) and tool_result.get("ok") is True:
                    is_success = True
        except Exception:
            is_success = False

        if is_success:
            next_idx = idx + 1
            try:
                debug.note("step_advanced", f"{idx} → {next_idx}")
            except Exception:
                pass
        else:
            next_idx = idx
            try:
                debug.note("step_not_advanced", f"未得到 success 判定，保持在 {idx}")
            except Exception:
                pass
    elif observer_route == "plan":
        # 交由 plan 重新规划，索引按需要可保持
        next_idx = idx
    elif observer_route == "end":
        next_idx = idx

    # 如果工具返回了新的 session_token（会话被新建或重建），更新到状态
    new_token = None
    try:
        if isinstance(tool_result, dict):
            new_token = tool_result.get("session_token")
    except Exception:
        pass

    # 结束判定完全由 Observer 决定
    steps_len = len(task.get("steps", []))
    is_complete = False
    failed = False
    
    # 只有 Observer 明确返回 "end" 才结束任务
    if observer_route == "end":
        is_complete = True
        try:
            debug.note("task_completed", f"Observer 决定结束任务")
        except Exception:
            pass
    # 初始化路由状态
    new_mode = state.get("mode", "discover")
    new_episode = int(state.get("episode", 1) or 1)
    replan_requested = False
    next_route = "decide"
    
    # 优先级 1：检查模式切换（discover ↔ execute）
    # 模式切换会触发重规划，优先级最高
    try:
        suggested_mode = route_decision.get("mode")
        if suggested_mode in ("discover", "execute") and suggested_mode != new_mode:
            # 防止频繁重规划：同一 episode 内只允许模式切换触发一次重规划
            last_replan_episode = state.get("_last_replan_episode", 0)
            if new_episode != last_replan_episode:
                new_mode = suggested_mode
                new_episode = new_episode + 1
                replan_requested = True
                next_route = "plan"
                try:
                    debug.note("mode_switch_triggered", f"{state.get('mode', 'discover')} -> {suggested_mode}, episode {new_episode}")
                except Exception:
                    pass
            else:
                try:
                    debug.note("mode_switch_throttled", f"同一 episode 内已触发过模式切换，忽略本次建议")
                except Exception:
                    pass
    except Exception:
        pass
    
    # 优先级 2：处理 Observer 的路由决策
    # 只有在没有触发模式切换的情况下才处理路由
    if not replan_requested:
        if is_complete:
            # 任务完成，结束工作流
            next_route = END
        elif observer_route == "plan":
            replan_requested = True
            next_route = "plan"
        elif observer_route == "decide":
            next_route = "decide"
        elif observer_route == "repeat_step":
            next_route = "decide"
        elif observer_route == "skip_step":
            next_route = "decide"
        elif observer_route == "end":
            next_route = END

    # 在返回前对 facts 进行规范化（绝对路径、占位符展开）
    try:
        from utils import normalize_facts
        facts = normalize_facts(facts)
    except Exception:
        pass

    # 简单的重复上限：当观察器建议 repeat_step 多于 2 次时，转为 plan
    try:
        repeat_counts = dict(state.get("repeat_counts", {}))
        key = str(task.get("steps", [])[idx].get("title", idx)) if 0 <= idx < len(task.get("steps", [])) else str(idx)
        if observer_route == "repeat_step":
            repeat_counts[key] = int(repeat_counts.get(key, 0)) + 1
            debug.note("repeat_count", f"步骤 {idx} 重复次数: {repeat_counts[key]}")
            if repeat_counts[key] > 2:
                # 重复太多次，触发重规划
                replan_requested = True
                next_route = "plan"
                debug.note("repeat_limit_exceeded", f"步骤 {idx} 重复超过 2 次，触发重规划")
        else:
            # 非重复则重置计数
            repeat_counts[key] = 0
    except Exception:
        repeat_counts = state.get("repeat_counts", {})

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
        "repeat_counts": repeat_counts,
        # 记录本次重规划的 episode，用于防止频繁重规划
        "_last_replan_episode": new_episode if replan_requested else state.get("_last_replan_episode", 0),
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
    """构建任务执行 LangGraph 工作流（discover → plan → decide → execute → observe → (END|plan|decide)）。"""
    workflow = StateGraph(AgentState)
    workflow.add_node("discover", discover_node)
    workflow.add_node("plan", plan_node)
    # 决策节点：决定下一步调用的工具或是否重规划
    workflow.add_node("decide", decide_node)
    # ToolNode: 暴露执行与文件系统只读工具
    workflow.add_node("execute", ToolNode([
        RUN_INSTRUCTION_TOOL,
        FILES_EXISTS_TOOL,
        FILES_STAT_TOOL,
        FILES_LIST_TOOL,
        FILES_READ_TOOL,
        FILES_FIND_TOOL,
        PYENV_PYTHON_INFO_TOOL,
        PYENV_TOOL_VERSIONS_TOOL,
        PYENV_PARSE_PYPROJECT_TOOL,
        PYENV_SELECT_INSTALLER_TOOL,
        GIT_REPO_STATUS_TOOL,
        GIT_ENSURE_CLONED_TOOL,
    ]))
    workflow.add_node("observe", observe_node)

    workflow.set_entry_point("discover")
    workflow.add_edge("discover", "plan")
    workflow.add_edge("plan", "decide")
    
    # decide节点的条件路由：根据决策结果决定是执行工具还是重规划
    def _decide_route(s: AgentState):
        if s.get("replan_requested") or s.get("route") == "plan":
            return "plan"
        return "execute"
    
    workflow.add_conditional_edges(
        "decide",
        _decide_route,
        {"plan": "plan", "execute": "execute"},
    )
    
    workflow.add_edge("execute", "observe")
    
    # 根据观察路由决定流向：END / plan(重规划) / decide(继续)
    def _observe_route(s: AgentState):
        if s.get("is_complete"):
            return END
        route = s.get("route")
        if route == "plan" or s.get("replan_requested"):
            return "plan"
        # 默认回到决策节点继续推进
        return "decide"

    workflow.add_conditional_edges(
        "observe",
        _observe_route,
        {END: END, "plan": "plan", "decide": "decide"},
    )

    return workflow.compile()

