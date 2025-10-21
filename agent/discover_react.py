from __future__ import annotations

from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from utils import llm_completion
from agent.debug import dispInfo, debug
from agent.message_utils import make_generic_tool_call_message, extract_last_tool_result
from tools import (
    FILES_EXISTS_TOOL,
    FILES_STAT_TOOL,
    FILES_LIST_TOOL,
    FILES_READ_TOOL,
    FILES_FIND_TOOL,
    FILES_READ_SECTION_TOOL,
    FILES_READ_RANGE_TOOL,
    FILES_GREP_TOOL,
    MD_OUTLINE_TOOL,
    PYENV_PYTHON_INFO_TOOL,
    PYENV_TOOL_VERSIONS_TOOL,
    PYENV_PARSE_PYPROJECT_TOOL,
    GIT_REPO_STATUS_TOOL,
)


class DiscoverState(TypedDict, total=False):
    """State for the standalone ReAct discover agent.

    Fields:
    - goal: discovery goal text
    - messages: LangChain-style message list for ToolNode routing
    - transcript: list of {thought, action, observation} for human-readable trace
    - summary: final natural-language plan/understanding
    - finished: whether the agent decided to stop
    - facts: optional scratch facts accumulated during discovery (best-effort)
    - route: internal routing hint for graph edges
    """

    goal: str
    messages: List[dict]
    transcript: List[Dict[str, Any]]
    summary: str
    finished: bool
    facts: Dict[str, Any]
    route: str


SYSTEM_PROMPT = (
    "你是一个只读的项目侦察代理。你的任务是逐步理解工作区中的项目结构，"
    "并在自认为信息充分时给出清晰的\"安装与运行方案\"总结。\n\n"
    "规则：\n"
    "- 只使用以下已注册的只读工具：files_list, files_read, files_read_section, files_read_range, files_grep, md_outline, files_exists, files_find, files_stat, "
    "pyenv_python_info, pyenv_tool_versions, pyenv_parse_pyproject, git_repo_status\n"
    "- 严禁执行修改性操作（不得调用 run_instruction、git_ensure_cloned 等）\n"
    "- 路径参数使用相对路径或 'repo_root' 占位符（工具会自动解析为工作区根目录）\n"
    "- **策略**：先 files_list 获取全貌 → 根据 list 结果中的\"关键文件\"决定后续步骤 → 避免盲目试探\n"
    "- **关键点**：files_list 的 Observation 会突出显示 pyproject.toml/setup.py/requirements.txt 等关键文件，请优先关注\n"
    "- **结束条件**：获得依赖信息（pyproject 或 requirements.txt）+ 可选的 README 关键信息后，即可 finish\n\n"
    "对话格式：\n"
    "Thought: 你的推理\n"
    "Action: <tool_name>(arg1=value1, arg2=value2)  或  finish\n\n"
    "路径示例：\n"
    "- files_list(path=\"repo_root\")  # 列出工作区根目录（会突出显示关键文件）\n"
    "- files_read(path=\"repo_root/README.md\", mode=\"head\")  # 读取 README 头部\n"
    "- md_outline(path=\"repo_root/README.md\")  # 提取 Markdown 目录，用于定位小节\n"
    "- files_read_section(path=\"repo_root/README.md\", start_line=20, end_line=80)  # 精准读取小节\n"
    "- pyenv_parse_pyproject(pyproject_path=\"pyproject.toml\")  # 解析 pyproject.toml（使用相对路径）\n" 
)


def _truncate(text: str, n: int = 1200) -> str:
    return (text or "")[:n]


def _build_react_prompt(state: DiscoverState) -> str:
    goal = state.get("goal", "")
    transcript = state.get("transcript", [])
    ctx_lines: List[str] = [SYSTEM_PROMPT]
    ctx_lines.append(f"Goal: {goal}\n")

    # Add short transcript context (last few turns)
    for turn in transcript[-6:]:
        th = str(turn.get("thought", "")).strip()
        ac = str(turn.get("action", "")).strip()
        ob = str(turn.get("observation", "")).strip()
        if th:
            ctx_lines.append(f"Thought: {th}")
        if ac:
            ctx_lines.append(f"Action: {ac}")
        if ob:
            # Observation is already smart and concise from observe_node
            ctx_lines.append(f"Observation: {_truncate(ob, 1500)}")

    ctx_lines.append("\n请基于以上信息继续：只输出一段 Thought 和一行 Action。")
    return "\n".join(ctx_lines)


def _parse_action(line: str) -> Dict[str, Any]:
    """Parse a lightweight action line like:
    Action: files_list(path="repo_root")
    Action: pyenv_parse_pyproject(pyproject_path="repo_root/pyproject.toml")
    Action: finish
    Returns {"type": "finish"} or {"type":"tool", "name": str, "args": dict}
    """
    text = (line or "").strip()
    if not text:
        return {"type": "invalid"}
    
    # Normalize prefix: remove multiple "Action:" prefixes
    while text.lower().startswith("action:"):
        text = text.split(":", 1)[1].strip()
    
    # Finish
    if text.lower().startswith("finish") or text.lower() in ("done", "no more actions"):
        return {"type": "finish"}

    # ToolName(args)
    import re
    m = re.match(r"([a-zA-Z_][\w]*)\s*\((.*)\)\s*$", text, re.DOTALL)
    if not m:
        return {"type": "invalid"}
    name = m.group(1)
    args_src = m.group(2).strip()
    kwargs: Dict[str, Any] = {}
    
    if args_src:
        # Parse kwargs manually to handle Windows paths better
        # Split by commas outside quotes
        import shlex
        try:
            # Use a simple key=value parser
            pairs = []
            current = ""
            in_quote = False
            quote_char = None
            
            for char in args_src:
                if char in ('"', "'") and (not in_quote or char == quote_char):
                    in_quote = not in_quote
                    quote_char = char if in_quote else None
                    current += char
                elif char == ',' and not in_quote:
                    if current.strip():
                        pairs.append(current.strip())
                    current = ""
                else:
                    current += char
            if current.strip():
                pairs.append(current.strip())
            
            # Parse each key=value pair
            for pair in pairs:
                if '=' not in pair:
                    continue
                key, val = pair.split('=', 1)
                key = key.strip()
                val = val.strip()
                
                # Evaluate value
                try:
                    # Try literal_eval for numbers, booleans, lists, etc.
                    import ast
                    kwargs[key] = ast.literal_eval(val)
                except Exception:
                    # Fallback: strip quotes if present, otherwise use as-is
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        kwargs[key] = val[1:-1]
                    else:
                        kwargs[key] = val
        except Exception:
            # If all parsing fails, return empty args
            kwargs = {}
    
    return {"type": "tool", "name": name, "args": kwargs}


@dispInfo("discover_react")
def react_node(state: DiscoverState) -> DiscoverState:
    """LLM-driven step: produce next Thought + Action. If Action is a tool call,
    enqueue a ToolNode request. If finish, route to summarize.
    """
    # Build prompt from transcript (observations already processed by observe_node)
    prompt = _build_react_prompt(state)
    
    # Debug: show prompt context (first time and every 5 turns)
    transcript = state.get("transcript", [])
    if len(transcript) == 0 or len(transcript) % 5 == 0:
        print(f"\n[Prompt Context - Turn {len(transcript)}]")
        print(f"Transcript entries: {len(transcript)}")
        if transcript:
            last = transcript[-1]
            print(f"Last action: {last.get('action', '')[:80]}")
            print(f"Last observation: {str(last.get('observation', ''))[:150]}...")

    model_out = llm_completion(prompt, temperature=0.2, max_tokens=400).strip()

    # Split out Thought and Action lines (best-effort)
    thought = ""
    action_line = ""
    for line in model_out.splitlines():
        if not thought and line.strip().lower().startswith("thought:"):
            thought = line.split(":", 1)[1].strip()
        if not action_line and line.strip().lower().startswith("action:"):
            action_line = line.strip()
    if not action_line:
        # Heuristic: try to find a tool-like pattern
        for line in model_out.splitlines():
            if "(" in line and ")" in line:
                action_line = "Action: " + line.strip()
                break

    # Print LLM decision
    print(f"\n{'='*80}")
    print(f"[LLM Decision]")
    print(f"Thought: {thought}")
    print(f"Action: {action_line}")
    print(f"{'='*80}")

    parsed = _parse_action(action_line)
    print(f"[Parsed] Type: {parsed.get('type')}, Name: {parsed.get('name')}, Args: {parsed.get('args')}")

    # Update transcript with Thought and Action (observation appended after tools run)
    transcript = list(state.get("transcript", []))
    transcript.append({"thought": thought, "action": action_line})

    messages = list(state.get("messages", []))
    route = "react"

    if parsed.get("type") == "finish":
        # Finish the loop
        return {
            **state,
            "messages": messages,
            "transcript": transcript,
            "route": "summarize",
        }

    if parsed.get("type") == "tool":
        tool_name = str(parsed.get("name", "")).strip()
        tool_args = parsed.get("args") or {}
        # Guardrail: only allow the whitelisted read-only tools
        allowed = {
            "files_exists",
            "files_stat",
            "files_list",
            "files_read",
            "files_find",
            "files_read_section",
            "files_read_range",
            "files_grep",
            "md_outline",
            "pyenv_python_info",
            "pyenv_tool_versions",
            "pyenv_parse_pyproject",
            "git_repo_status",
        }
        if tool_name in allowed:
            try:
                print(f"[Tool Call] {tool_name}({tool_args})")
                messages.append(make_generic_tool_call_message(tool_name, tool_args))
                route = "execute"
            except Exception:
                # Could not enqueue tool; record as observation and continue
                transcript[-1]["observation"] = f"enqueue_error: {tool_name}"
                route = "react"
        else:
            # Unsupported/mutating action: reflect and continue
            print(f"[Blocked] Unsupported action: {tool_name}")
            transcript[-1]["observation"] = f"unsupported_action: {tool_name}"
            route = "react"
    else:
        transcript[-1]["observation"] = "invalid_action"
        route = "react"

    return {**state, "messages": messages, "transcript": transcript, "route": route}


@dispInfo("discover_react")
def observe_node(state: DiscoverState) -> DiscoverState:
    """After ToolNode runs, capture the latest observation into transcript, then loop back."""
    transcript = list(state.get("transcript", []))
    last = extract_last_tool_result(state.get("messages", []))
    
    # Build a smart, concise observation for the LLM (not truncated)
    llm_observation = ""
    if last:
        import json
        from tools.base import parse_tool_response
        parsed = parse_tool_response(json.dumps(last) if isinstance(last, dict) else str(last))
        tool_name = parsed.get('tool', 'unknown')
        ok = parsed.get('ok', False)
        data = parsed.get('data', {})
        error = parsed.get('error')
        
        # Print full result for debugging
        print(f"[Tool Result]")
        print(f"  Tool: {tool_name}")
        print(f"  OK: {ok}")
        data_str = json.dumps(data, ensure_ascii=False)
        if len(data_str) > 300:
            print(f"  Data: {data_str[:300]}...")
        else:
            print(f"  Data: {data_str}")
        if error:
            print(f"  Error: {error}")
        
        # Build smart observation for LLM (full context, not truncated)
        if not ok:
            llm_observation = f"工具 {tool_name} 失败: {error or '未知错误'}"
        elif tool_name == "files_read":
            content = data.get("content", "")
            size = data.get("size", len(content))
            truncated = data.get("truncated", False)
            path = data.get("path", "")
            llm_observation = f"已读取文件 {path}（共 {size} 字符）{'[被截断]' if truncated else '[完整]'}:\n{content}"
        elif tool_name == "files_list":
            entries = data.get("entries", [])
            truncated = data.get("truncated", False)
            dir_path = data.get("dir", "")
            
            # Categorize entries for better readability 
            key_files = []  # Important config files
            py_files = []
            dirs = []
            others = []
            
            for e in entries:
                name = e.get("name", "")
                etype = e.get("type", "file")
                
                # Highlight key files
                if name.lower() in ("pyproject.toml", "setup.py", "requirements.txt", "readme.md", "readme", "setup.cfg"):
                    key_files.append(f"{name} [{etype}]")
                elif name.endswith(".py"):
                    py_files.append(name)
                elif etype == "dir":
                    dirs.append(name + "/")
                else:
                    others.append(name)
            
            # Build hierarchical observation
            parts = [f"目录 {dir_path} 包含 {len(entries)} 项:"]
            if key_files:
                parts.append(f"  关键文件: {', '.join(key_files)}")
            if dirs:
                parts.append(f"  子目录({len(dirs)}): {', '.join(dirs[:10])}{'...' if len(dirs) > 10 else ''}")
            if py_files:
                parts.append(f"  Python文件({len(py_files)}): {', '.join(py_files[:8])}{'...' if len(py_files) > 8 else ''}")
            if others and len(others) <= 15:
                parts.append(f"  其他: {', '.join(others)}")
            
            llm_observation = "\n".join(parts)
        elif tool_name == "files_exists":
            exists = data.get("exists", False)
            path = data.get("path", "")
            llm_observation = f"文件 {path} {'存在' if exists else '不存在'}"
        elif tool_name == "pyenv_parse_pyproject":
            exists = data.get("exists", False)
            if not exists:
                llm_observation = f"pyproject.toml 不存在于 {data.get('path', '')}"
            else:
                deps = data.get("dependencies", [])
                name = data.get("project_name", "")
                llm_observation = f"项目 {name}，依赖 {len(deps)} 个包: {', '.join(deps[:10])}{'...' if len(deps) > 10 else ''}"
        else:
            # Fallback: compact JSON
            llm_observation = json.dumps(data, ensure_ascii=False)[:800]
    
    if transcript:
        try:
            # Store smart observation for LLM, not the raw tool output
            transcript[-1]["observation"] = llm_observation
        except Exception:
            pass

    # Align facts handling with unified observer: compute facts_delta and normalize
    try:
        from agent.observer import observe_v2
        task_ctx = {"goal": state.get("goal", ""), "steps": [{"title": "discover"}]}
        route_decision = observe_v2(task_ctx, 0, last, mode="discover", episode=1, facts=state.get("facts", {}))
        facts = dict(state.get("facts", {}))
        try:
            delta = route_decision.get("facts_delta") or {}
            if isinstance(delta, dict):
                facts.update(delta)
        except Exception:
            pass
        # Normalize minimal expected keys
        if "repo_path" in facts and not facts.get("repo_root"):
            facts["repo_root"] = facts.pop("repo_path")
        if facts.get("repo_root") and not facts.get("exec_root"):
            facts["exec_root"] = facts["repo_root"]
        facts.pop("work_dir", None)
        state = {**state, "facts": facts}
    except Exception:
        # facts extraction is best-effort; ignore failures
        pass

    return {**state, "transcript": transcript, "route": "react"}


@dispInfo("discover_react")
def summarize_node(state: DiscoverState) -> DiscoverState:
    """Ask the model to produce a natural-language understanding & setup plan."""
    transcript = state.get("transcript", [])
    goal = state.get("goal", "")

    # Build a compact transcript for the model
    lines: List[str] = [
        "你已经完成对项目的只读探索。请根据以下简要对话撰写总结：",
        f"目标: {goal}",
        "\n对话片段 (节选):",
    ]
    for t in transcript[-12:]:
        th = str(t.get("thought", "")).strip()
        ac = str(t.get("action", "")).strip()
        ob = t.get("observation")
        ob_str = _truncate(str(ob)) if ob is not None else ""
        if th:
            lines.append(f"Thought: {th}")
        if ac:
            lines.append(f"Action: {ac}")
        if ob_str:
            lines.append(f"Observation: {ob_str}")

    lines.append(
        (
            "\n请输出一个清晰、实用的“项目理解与安装/运行方案”（要点式），"
            "包含：\n- 关键信息（项目类型/依赖管理方式/可能的运行入口）\n"
            "- 安装建议（选择 uv/pip/poetry/pdm/conda 的理由与示例命令）\n"
            "- 运行/测试建议（可能的命令、注意事项）\n"
            "- 若信息不足，注明缺口与下一步建议（阅读哪些文件）\n"
        )
    )

    try:
        summary = llm_completion("\n".join(lines), temperature=0.2, max_tokens=700).strip()
    except Exception:
        summary = "(总结失败)"

    return {**state, "summary": summary, "finished": True}


def _react_route(state: DiscoverState):
    route = state.get("route", "react")
    if route == "execute":
        return "execute"
    if route == "summarize":
        return "summarize"
    return "react"


def create_discover_graph():
    """Build a standalone discover graph: react → execute(ToolNode) → observe → react ... → summarize → END"""
    g = StateGraph(DiscoverState)

    # LLM thought/action node
    g.add_node("react", react_node)

    # Tool execution node (read-only tools only)
    g.add_node(
        "execute",
        ToolNode(
            [
                FILES_EXISTS_TOOL,
                FILES_STAT_TOOL,
                FILES_LIST_TOOL,
                FILES_READ_TOOL,
                FILES_FIND_TOOL,
                FILES_READ_SECTION_TOOL,
                FILES_READ_RANGE_TOOL,
                FILES_GREP_TOOL,
                MD_OUTLINE_TOOL,
                PYENV_PYTHON_INFO_TOOL,
                PYENV_TOOL_VERSIONS_TOOL,
                PYENV_PARSE_PYPROJECT_TOOL,
                GIT_REPO_STATUS_TOOL,
            ]
        ),
    )

    # Observation assimilation node
    g.add_node("observe", observe_node)

    # Final summarization node
    g.add_node("summarize", summarize_node)

    # Entry and edges
    g.set_entry_point("react")
    g.add_conditional_edges("react", _react_route, {"execute": "execute", "summarize": "summarize", "react": "react"})
    g.add_edge("execute", "observe")
    g.add_edge("observe", "react")
    g.add_edge("summarize", END)

    return g.compile()


def run_discover_react(goal: str, seed_facts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the standalone discover agent to build understanding and produce a summary.

    Returns a dict with keys: transcript, summary, facts (best-effort), raw_state
    """
    app = create_discover_graph()
    init: DiscoverState = {
        "goal": goal,
        "messages": [],
        "transcript": [],
        "summary": "",
        "finished": False,
        "facts": dict(seed_facts or {}),
        "route": "react",
    }
    try:
        out = app.invoke(init, {"recursion_limit": 100})
    except Exception as e:
        try:
            debug.note("discover_react_error", str(e))
        except Exception:
            pass
        out = init
        out["summary"] = f"(运行失败) {type(e).__name__}: {e}"

    return {
        "transcript": out.get("transcript", []),
        "summary": out.get("summary", ""),
        "facts": out.get("facts", {}),
        "raw_state": out,
    }


