from typing import Optional, Dict, Any
from agent.task_types import Task
from utils import llm_completion
from agent.debug import dispInfo, debug


def _short(text: Optional[str], n: int = 600) -> str:
    return (text or "")[:n]


# @dispInfo("decider")
def summarize_context(task: Task, current_index: int, last_result: Optional[Dict[str, Any]], *, mode: str = "", episode: int = 0, facts: Optional[Dict[str, Any]] = None) -> str:
    """汇总上下文（目标/计划/当前位置/上一次结果），用于喂给LLM。"""
    steps = task.get("steps", [])
    plan_titles = [s.get("title", f"步骤{i+1}") for i, s in enumerate(steps)]
    current_step = steps[current_index] if 0 <= current_index < len(steps) else None

    import json as _json
    from tools.base import parse_tool_response
    
    # 格式化上一次执行结果（使用统一工具接口）
    last_result_str = ""
    if last_result:
        parsed = parse_tool_response(_json.dumps(last_result))
        tool_name = parsed.get("tool", "unknown")
        tool_ok = parsed.get("ok", False)
        tool_data = parsed.get("data", {})
        tool_error = parsed.get("error")
        
        # 根据工具类型显示关键数据
        if tool_name == "run_instruction":
            exit_code = tool_data.get("exit_code")
            command = tool_data.get("command", "")
            stdout = _short(tool_data.get("stdout", ""), 400) 
            stderr = _short(tool_data.get("stderr", ""), 400)
            last_result_str = (
                f"工具: {tool_name}\n"
                f"命令: {command}\n"
                f"退出码: {exit_code}\n"
                f"STDOUT: {stdout}\n"
                f"STDERR: {stderr}\n"
                f"错误: {tool_error or '无'}"
            )
        else:
            # 其他工具显示关键字段
            key_fields = ["path", "exists", "content", "type", "dir", "installer", "reason"]
            key_data = {k: v for k, v in tool_data.items() if k in key_fields}
            last_result_str = (
                f"工具: {tool_name}\n"
                f"状态: {'成功' if tool_ok else '失败'}\n"
                f"关键数据: {_json.dumps(key_data, ensure_ascii=False)[:400]}\n"
                f"错误: {tool_error or '无'}"
            )
    
    return (
        f"任务目标: {task.get('goal','')}\n"
        f"计划步骤（标题序列）: {plan_titles}\n"
        f"当前步骤索引: {current_index}\n"
        f"当前步骤详情: {current_step}\n"
        f"当前模式: {mode} 周期: {episode}\n"
        f"已知事实: {_json.dumps(facts or {}, ensure_ascii=False)[:1000]}\n"
        f"上一次结果:\n{last_result_str if last_result_str else '(无)'}\n"
    )


# @dispInfo("decider")
def decide_next_action(task: Task, current_index: int, last_result: Optional[Dict[str, Any]], *, mode: str = "", episode: int = 0, facts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    决策下一步操作：
    返回结构：{"action": "call_tool"|"replan", "nl_instruction": str?, "timeout": int?, "session_token": str?}
    """
    context = summarize_context(task, current_index, last_result, mode=mode, episode=episode, facts=facts)
    try:
        debug.note("context_summary", context[:600])
    except Exception:
        pass

    prompt = (
        "你是任务执行的指挥官，根据计划步骤的**意图描述**与执行结果，严格依据 facts 优先级进行决策，优先调用专用工具，仅在无法覆盖的情况下使用通用命令。\n\n"
        
        "【可用工具清单（优先使用专用工具）】\n"
        "一、文件系统工具（discover 模式优先）：\n\n"
        
        "1. files_exists - 检查文件/目录是否存在\n"
        "   参数: {\"path\": \"相对路径或绝对路径\"}\n"
        "   适用意图: 检查/探测/验证 XX 是否存在\n"
        "   示例: \"检查transformers目录是否存在\" → files_exists\n\n"
        
        "2. files_list - 列出目录内容\n"
        "   参数: {\"path\": \"目录路径\", \"recurse\": false, \"files_only\": false, \"patterns\": []}\n"
        "   适用意图: 列出/查看/浏览 XX 目录的文件\n"
        "   示例: \"列出setupLab2目录下的所有文件\" → files_list\n\n"
        
        "3. files_read - 读取文件内容\n"
        "   参数: {\"path\": \"文件路径\", \"mode\": \"raw\"}\n"
        "   适用意图: 读取/查看/获取 XX 文件的内容\n"
        "   示例: \"读取pyproject.toml文件的内容\" → files_read\n\n"
        
        "4. files_find - 搜索文件\n"
        "   参数: {\"start_dir\": \"起始目录\", \"include_globs\": [\"*.py\"], \"first_only\": false}\n"
        "   适用意图: 查找/搜索/定位 XX 文件\n"
        "   示例: \"查找所有Python文件\" → files_find\n\n"
        
        "二、Python 环境工具：\n\n"
        
        "5. pyenv_python_info - 探测 Python 解释器\n"
        "   参数: {}\n"
        "   适用意图: 检测/探测/查看 Python 版本或解释器\n\n"
        
        "6. pyenv_parse_pyproject - 解析 pyproject.toml\n"
        "   参数: {\"pyproject_path\": \"路径\"}\n"
        "   适用意图: 解析/分析 pyproject.toml 配置\n\n"
        
        "7. pyenv_select_installer - 选择包管理器\n"
        "   参数: {\"project_root\": \"路径\"}\n"
        "   适用意图: 选择/推荐安装器（uv/pip/poetry等）\n\n"
        
        "三、Git 工具（优先用于仓库相关意图）：\n\n"
        "8. git_repo_status - 检查目录是否为 Git 仓库并返回 origin/分支等\n"
        "   参数: {\"path\": \"目录路径\"}\n"
        "   适用意图: 确认仓库是否存在并可用/获取 origin 与分支\n\n"
        "9. git_ensure_cloned - 确保仓库已在工作区可用（若不存在则浅克隆）\n"
        "   参数: {\"url\": \"仓库URL\", \"dest\": \"可选目标路径\", \"depth\": 1, \"sparse\": true, \"branch\": \"可选\"}\n"
        "   适用意图: 确保仓库已可用/若不存在则克隆（避免重复克隆）\n"
        "   重要规则: 不要把 '.' 作为 dest 传入；如无特定目标目录，省略 dest 即可\n\n"

        "四、通用执行工具（兜底，仅在专用工具不适用时使用）：\n\n"
        
        "8. run_instruction - 执行任意 shell 命令\n"
        "   用于: git clone、pip install、运行脚本等所有命令行操作\n"
        "   适用意图: 所有非探测性的执行操作\n\n"

        "【Facts 优先级（必须遵守）】\n"
        "- 任何决策都必须首先检查并遵守 facts；不得生成与 facts 冲突的操作。\n"
        "- 若 facts.project_root 已存在：禁止产生“克隆仓库/重复探测工作区根目录”的行为，应直接在项目目录开展后续步骤。\n"
        "- 若 facts.has_pyproject=true：解析/读取应以 facts.project_root 下的文件为准，不要对工作区根发起 pyproject 解析。\n"
        "- 若 facts.has_readme/has_setup_py/has_requirements_txt 已知：避免重复探测这些文件是否存在。\n"
        "- Git 相关意图由下游映射到结构化 git 工具（状态/确保已克隆），避免直接 shell 级 git 指令。\n\n"

        "【安装命令路径规范】\n"
        "- 在项目目录执行的命令，一律使用 $env:PROJECT_ROOT。\n"
        "- 可编辑安装示例: pip install -e $env:PROJECT_ROOT\n"
        "- 需要进入目录执行时: cd $env:PROJECT_ROOT; <后续命令>。\n"
        "- 禁止使用 (Join-Path $env:REPO_ROOT 'xxx') 这类固定拼接，避免与 REPO_ROOT 变更产生偏差。\n\n"
        
        "【路径参数规范（极重要）】\n"
        "- 工作区根（REPO_ROOT）已设置到环境变量，files_* 工具会将相对路径自动解析为 REPO_ROOT 下的绝对路径。\n"
        "- 若要表示 REPO_ROOT 自身，请使用 \".\" 或空字符串。例：列出工作区根 → files_list {path: \".\"}\n"
        "- 严禁在 path 中重复拼接目录名（例如已有 REPO_ROOT=D:\\0APython\\setupLab2 时，不要再传入 \"setupLab2\\...\"）。\n"
        "- 严禁把 facts.project_root 之类的键名当作字面路径传入。需要路径就直接传入字符串路径（相对 REPO_ROOT）。\n"
        "- 允许使用绝对路径，但应尽量使用相对路径以保持可移植性。\n\n"

        "【决策规则】\n"
        "根据意图关键词和当前模式选择工具：\n\n"
        
        "discover 模式（探测阶段）：\n"
        "- 意图含\"列出/查看目录\" → files_list\n"
        "- 意图含\"检查/验证是否存在\" → files_exists\n"
        "- 意图含\"读取文件内容\" → files_read\n"
        "- 意图含\"查找/搜索文件\" → files_find\n"
        "- 意图含\"探测Python\" → pyenv_python_info\n"
        "- 意图含\"解析pyproject\" → pyenv_parse_pyproject\n"
        "- 意图含\"仓库/origin/分支/状态\" → git_repo_status\n\n"
        
        "execute 模式（执行阶段）：\n"
        "- 意图含\"确保仓库已可用/克隆\" → git_ensure_cloned\n"
        "- 意图含\"安装依赖/pip install\" → run_instruction（pip install ...）\n"
        "- 意图含\"运行/执行脚本\" → run_instruction\n"
        "- 其他所有命令行操作 → run_instruction\n\n"
        
        "特殊情况：\n"
        "- 上一步失败且需要小调整 → 修改命令后用 run_instruction\n"
        "- 上一步失败且需要大改动 → replan\n"
        "- 意图不清晰或无法判断 → run_instruction（兜底）\n\n"
        
        f"{context}\n\n"
        
        "【输出格式】严格的 JSON，不要多余文字：\n\n"
        
        "调用专用工具时：\n"
        '{\"action\": \"call_tool\", \"tool_name\": \"files_list\", \"tool_args\": {\"path\": \"transformers\"}}\n\n'
        
        "执行 shell 命令时：\n"
        '{\"action\": \"call_instruction\", \"nl_instruction\": \"克隆仓库到工作目录\", \"timeout\": 300}\n\n'
        
        "请求重新规划时：\n"
        '{\"action\": \"replan\"}\n\n'
        
        "注意事项：\n"
        "- call_tool 时必须提供 tool_name 和 tool_args\n"
        "- call_instruction 时必须提供 nl_instruction（可以是意图描述或具体命令）\n"
        "- git clone/pull 等耗时操作设置 timeout >= 300\n"
        "- 必须优先使用专用工具而不是 run_instruction；run_instruction 仅作为兜底选项\n\n"
        
        "根据当前步骤的意图描述，输出你的决策（仅 JSON）："
    )

    # debug.note("decide_prompt", prompt)  # 提示词太长，不记录
    try:
        debug.note("prompt_length", len(prompt))
    except Exception:
        pass
    
    try:
        resp = llm_completion(prompt, temperature=0.2, max_tokens=300).strip()
    except Exception as e:
        try:
            debug.note("llm_error", f"{type(e).__name__}: {str(e)[:200]}")
        except Exception:
            pass
        resp = "{}"
    
    try:
        debug.note("decide_raw_resp_len", len(resp))
    except Exception:
        pass
    debug.note("decide_raw_resp", resp)

    import json, re
    data: Optional[Dict[str, Any]] = None
    try:
        data = json.loads(resp)
    except Exception as e:
        try:
            debug.note("json_parse_error", f"{type(e).__name__}: {str(e)[:100]}")
        except Exception:
            pass
        m = re.search(r'\{.*\}', resp, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None

    if not isinstance(data, dict):
        try:
            debug.note("fallback_to_instruction", "LLM返回无效，使用回退逻辑")
        except Exception:
            pass
        # 回退：无有效决策时，尝试继续按计划执行
        steps = task.get("steps", [])
        step = steps[current_index] if 0 <= current_index < len(steps) else None
        if not step:
            return {"action": "replan", "raw": resp}
        instruction = str(step.get("instruction", "")).strip()
        # 根据指令类型智能设置超时：git clone/pull 默认更长（1800秒）
        default_timeout = 1800 if ("git clone" in instruction.lower() or "git pull" in instruction.lower()) else 60
        return {
            "action": "call_instruction",
            "nl_instruction": instruction,
            "timeout": int(step.get("timeout") or default_timeout),
            "raw": resp,
        }

    action = data.get("action")
    if action == "replan":
        return {"action": "replan", "raw": resp}

    # 获取超时值，如果是git clone等操作默认使用更长的超时
    nl_instruction = str(data.get("nl_instruction", "")).strip()
    default_timeout = 60
    if nl_instruction and ("git clone" in nl_instruction.lower() or "git pull" in nl_instruction.lower()):
        default_timeout = 1800
    timeout = int(data.get("timeout") or default_timeout)
    session_token = data.get("session_token")

    # 分支：call_tool / call_instruction
    if action == "call_tool":
        tool_name = str(data.get("tool_name", "")).strip()
        tool_args = data.get("tool_args") or {}
        if not tool_name:
            return {"action": "replan", "raw": resp}
        return {"action": "call_tool", "tool_name": tool_name, "tool_args": tool_args, "timeout": timeout, "session_token": session_token, "raw": resp}

    # 默认为 call_instruction
    nl_instruction = str(data.get("nl_instruction", "")).strip()
    if not nl_instruction:
        # 回退计划步
        steps = task.get("steps", [])
        step = steps[current_index] if 0 <= current_index < len(steps) else None
        if not step:
            return {"action": "replan", "raw": resp}
        nl_instruction = str(step.get("instruction", "")).strip()
        # 根据指令类型智能设置超时
        default_timeout = 1800 if ("git clone" in nl_instruction.lower() or "git pull" in nl_instruction.lower()) else 60
        timeout = int(step.get("timeout") or default_timeout)
        if isinstance(last_result, dict):
            session_token = last_result.get("session_token") or session_token
    return {"action": "call_instruction", "nl_instruction": nl_instruction, "timeout": timeout, "session_token": session_token, "raw": resp}
