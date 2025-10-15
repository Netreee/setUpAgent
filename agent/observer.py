from typing import Tuple, Dict, Any, Optional
from agent.task_types import Task, StepResult
from utils import llm_completion
from agent.debug import dispInfo, debug


def _short(text: Optional[str], n: int = 600) -> str:
    return (text or "")[:n]


def observe_v2(task: Task, current_index: int, last_result: Optional[StepResult], *, mode: str, episode: int, facts: Dict[str, Any]) -> Dict[str, Any]:
    """LLM 驱动的观察与路由，返回结构化决策。"""
    steps = task.get("steps", [])
    titles = [s.get("title", f"步骤{i+1}") for i, s in enumerate(steps)]
    import json as _json
    import os as _os
    
    # 使用统一工具接口解析结果
    from tools.base import parse_tool_response
    
    try:
        _lr = dict(last_result or {})
    except Exception:
        _lr = {}
    
    # 解析工具返回
    parsed = parse_tool_response(_json.dumps(_lr))
    tool_name = parsed.get("tool", "unknown")
    tool_ok = parsed.get("ok", False)
    tool_data = parsed.get("data", {})
    tool_error = parsed.get("error")
    
    # 构造关键数据摘要（仅显示重要字段）
    key_fields = [
        "path", "exists", "content", "exit_code", "command", "installer", "reason",
        # git/files 关键字段补充
        "existed", "cloned", "project_root", "project_name", "repo_root",
        "remote_url", "branch", "is_repo", "dir", "entries", "truncated"
    ]
    data_summary = {k: v for k, v in tool_data.items() if k in key_fields}
    data_summary_str = _json.dumps(data_summary, ensure_ascii=False)[:300]
    
    # 提取 stdout/stderr（如果是 run_instruction）
    _stdout_full = str(tool_data.get("stdout", ""))
    _stderr_full = str(tool_data.get("stderr", ""))
    _stdout_len = len(_stdout_full)
    _stderr_len = len(_stderr_full)
    _slice = 600
    _stdout_head = _stdout_full[:_slice]
    _stdout_tail = _stdout_full[-_slice:] if _stdout_len > _slice else ""
    _stderr_head = _stderr_full[:_slice]
    _stderr_tail = _stderr_full[-_slice:] if _stderr_len > _slice else ""

    prompt = (
        "你是观察者。基于结果、模式与事实，决定下一跳路由与可能的事实增量。\n"
        f"模式: {mode} 周期: {episode}\n"
        f"目标: {task.get('goal','')}\n"
        f"计划步骤标题序列: {titles}\n"
        f"当前索引: {current_index}\n"
        f"最近一次结果（统一格式）：\n"
        f"  工具: {tool_name}\n"
        f"  状态: {'成功' if tool_ok else '失败'}\n"
        f"  关键数据: {data_summary_str}\n"
        f"  错误: {tool_error or '无'}\n"
        f"  输出长度: stdout={_stdout_len}字节, stderr={_stderr_len}字节\n"
        f"  stdout_head: {_short(_stdout_head)}\n"
        f"  stdout_tail: {_short(_stdout_tail)}\n"
        f"  stderr_head: {_short(_stderr_head)}\n"
        f"  stderr_tail: {_short(_stderr_tail)}\n"
        f"已知事实: {_json.dumps(facts or {}, ensure_ascii=False)[:1200]}\n\n"
        "【路由职责】\n"
        "- 你的主要职责是评估当前步骤的执行结果，决定步骤级路由\n"
        "- 路由集合: decide | repeat_step | skip_step | end\n"
        "- 仅在**极少数情况**下使用 route: \"plan\"（完全重规划）\n\n"
        "【模式切换规则】\n"
        "- discover 模式：收集项目信息（克隆、列目录、读取配置文件）\n"
        "- execute 模式：执行安装/构建/运行操作\n"
        "- 切换到 execute 的条件（满足以下**任一组合**即可）：\n"
        "  组合 A（标准 Python 项目）：\n"
        "    1. 已确认 project_root 存在（facts 中有 project_root）\n"
        "    2. 已确认至少一个依赖声明文件（has_setup_py=true 或 has_pyproject=true）\n"
        "  组合 B（有 requirements.txt）：\n"
        "    1. 已确认 project_root 存在\n"
        "    2. 已确认 requirements.txt 存在\n"
        "  组合 C（README 明确说明）：\n"
        "    1. 已确认 project_root 存在\n"
        "    2. 已读取 README 且其中包含明确的安装命令\n"
        "- **关键判断点**：\n"
        "  * 当读取 setup.py 或 pyproject.toml 后，如果文件内容包含依赖信息，应切换到 execute\n"
        "  * 不要等待读取 README，setup.py/pyproject.toml 本身就足够了\n"
        "  * 如果当前步骤是计划中的最后一个 discover 步骤，应主动切换模式\n"
        "- 模式切换会触发重规划，但这是必要的流程，不要过度犹豫\n\n"
        "【facts_delta 规范与逻辑推理】\n"
        "- 统一命名：仅使用 repo_root, project_root, project_name, exec_root 等标准键\n"
        "- 不要输出 clone_path/repo_path/work_dir 等旧键\n"
        "- 若克隆完成，请提供 project_root 与 project_name\n"
        "- 若已知 repo_root，请补充 exec_root = repo_root\n"
        "- 路径叙述一律以 repo_root/project_root 为参照，不使用绝对盘符路径\n\n"
        "【关键逻辑推理规则】\n"
        "1. **目录状态判断**：\n"
        "   - 如果 files_list 返回 entries=[] → work_dir_empty=true\n"
        "   - 如果 files_list 返回 entries=[...] 且数组不为空 → work_dir_empty=false\n"
        "   - **重要**：看到目标仓库目录已存在时，必须设置 project_root 指向该目录\n"
        "   - **示例推理**：entries包含 'Real-Time-Voice-Cloning' → work_dir_empty=false, project_root='repo_root/Real-Time-Voice-Cloning'\n"
        "2. **仓库状态推理**：\n"
        "   - 如果 git clone 报错 'already exists' → 仓库已存在，设置 project_root\n"
        "   - **推理过程**：看到错误信息 → 分析原因 → 更新facts → 决定下一步\n"
        "   - 如果看到目录列表包含目标仓库名 → 仓库已存在，更新 project_root\n"
        "3. **文件存在性判断**：\n"
        "   - files_exists 返回 exists=true → has_xxx=true\n"
        "   - files_exists 返回 exists=false → has_xxx=false\n"
        "   - 读取文件成功 → 对应的 has_xxx=true, xxx_read=true\n"
        "4. **逻辑一致性检查**：\n"
        "   - 在设置facts_delta前，检查与已有facts的一致性\n"
        "   - 如果发现矛盾，优先相信最新的工具结果\n"
        "   - **示例**：如果工具显示目录不为空，但facts中work_dir_empty=true，应更新为false\n\n"
        "【输出格式】\n"
        "仅输出 JSON：{\n"
        "  \"route\": \"decide|repeat_step|skip_step|end\",\n"
        "  \"mode\": \"discover|execute\" | null,\n"
        "  \"facts_delta\": { } | null,\n"
        "  \"success\": true | false | null,\n"
        "  \"notes\": \"一句话原因\"\n"
        "}\n\n"
        "若无法判断，请将 success 设为 false，并说明需要哪些关键信息。"
    )
    # debug.note("observer_prompt", prompt)  # 提示词太长，不记录
    try:
        resp = llm_completion(prompt, temperature=0.2, max_tokens=400).strip()
    except Exception:
        resp = "{}"
    debug.note("observer_raw_resp", resp)

    import json, re
    data: Optional[Dict[str, Any]] = None
    try:
        data = json.loads(resp)
    except Exception:
        m = re.search(r"\{.*\}", resp, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None
    if not isinstance(data, dict):
        data = {"route": "decide", "notes": "默认继续"}

    # 统一并规范 facts_delta 字段（映射旧键、补齐标准键）
    try:
        fd = data.get("facts_delta")
        if isinstance(fd, dict):
            # clone_path -> project_root
            if "clone_path" in fd and not fd.get("project_root"):
                fd["project_root"] = fd.pop("clone_path")
            # repo_path -> repo_root（向后兼容）
            if "repo_path" in fd and not fd.get("repo_root"):
                fd["repo_root"] = fd.pop("repo_path")
            # 填充 project_name
            pr = fd.get("project_root")
            if isinstance(pr, str) and pr and not fd.get("project_name"):
                fd["project_name"] = _os.path.basename(pr.rstrip("\\/")) or fd.get("project_name")
            # 补齐 exec_root = repo_root
            if not fd.get("exec_root"):
                if fd.get("repo_root"):
                    fd["exec_root"] = fd["repo_root"]
                elif isinstance(facts, dict) and facts.get("repo_root"):
                    fd["exec_root"] = facts.get("repo_root")
            # 若仅有 repo_root 与 project_name，尝试推导 project_root（纯字符串拼接）
            if not fd.get("project_root") and fd.get("repo_root") and fd.get("project_name"):
                try:
                    fd["project_root"] = _os.path.join(fd["repo_root"], fd["project_name"])  # 不访问文件系统
                except Exception:
                    pass
            # 清理旧键
            for legacy in ("work_dir",):
                if legacy in fd:
                    fd.pop(legacy, None)
            data["facts_delta"] = fd
    except Exception:
        pass
    return data


def extract_readme_info(readme_text: str) -> Dict[str, Any]:
    """
    todo 这一部分可以抽象成工具
    从 README 文本中提取结构化信息，返回字典。
    目标字段（可能的示例）：
    - project_name: 项目/包名
    - description: 简短描述
    - install_cmds: 安装命令列表（pip/pipx/conda/poetry/pdm）
    - run_cmds: 运行/启动命令列表（python -m, uvicorn, streamlit, pytest 等）
    - requirements: 关键依赖或最低 Python 版本
    - entry_points: 入口（命令/模块）
    - links: 外链（文档/主页/issue）
    """
    info: Dict[str, Any] = {}
    text = (readme_text or "").strip()
    if not text:
        return info

    import re
    # 粗略提取项目名（首行标题）
    m = re.search(r"^\s*#\s+(.+)$", text, re.MULTILINE)
    if m:
        info["project_name"] = m.group(1).strip()

    # 简短描述：标题下的第一段非空文本
    try:
        lines = text.splitlines()
        desc = []
        hit_title = False
        for line in lines:
            if not hit_title and re.match(r"^\s*#\s+", line):
                hit_title = True
                continue
            if hit_title:
                if line.strip() == "":
                    if desc:
                        break
                    else:
                        continue
                desc.append(line.strip())
        if desc:
            info["description"] = " ".join(desc)[:400]
    except Exception:
        pass

    # 命令提取
    install_cmds = re.findall(r"(?mi)^(?:\s*[-*]\s*)?(pip(?:x)?|conda|poetry|pdm)\s+[^\n]+$", text)
    run_cmds = re.findall(r"(?mi)^(?:\s*[-*]\s*)?(python\s+-m\s+\S+|pytest\b[^\n]*|uvicorn\b[^\n]*|streamlit\b[^\n]*|gunicorn\b[^\n]*|make\s+\S+)\s*$", text)
    if install_cmds:
        info["install_cmds"] = list({cmd.strip() for cmd in install_cmds})
    if run_cmds:
        info["run_cmds"] = list({cmd.strip() for cmd in run_cmds})

    # Python 版本/依赖的线索
    pyver = re.search(r"(?i)python\s*(?:>=|=>|>=\s*)?\s*([0-9]+\.[0-9]+)", text)
    if pyver:
        info["python_min_version"] = pyver.group(1)

    # 入口点线索
    entry_cmds = re.findall(r"(?mi)^(?:\s*[-*]\s*)?(?:Usage:|命令:)?\s*(\w[\w-]+)\b[^\n]*$", text)
    if entry_cmds:
        info["entry_points"] = list({c.strip() for c in entry_cmds})

    # 链接提取
    links = re.findall(r"\((https?://[^)]+)\)", text)
    if links:
        info["links"] = list({u for u in links})

    return info


# @dispInfo("observer")
def observe(task: Task, current_index: int, last_result: StepResult | None) -> Tuple[bool, bool, str]:
    """
    保留旧签名给工作流使用。推荐通过外层 workflow 注入 mode/episode/facts 并使用 observe_v2 的结果更新 state。
    这里返回值仅做日志用，不做硬路由判定。
    """
    steps_len = len(task.get("steps", []))
    if steps_len == 0:
        return True, False, "没有可执行的步骤，视为完成"
    if last_result is None and current_index == 0:
        return False, False, "尚未开始执行"
    if last_result is not None and last_result.get("exit_code", -1) != 0:
        # 失败时给一句话建议
        try:
            prompt = (
                "请用一句话概括失败的原因与修复方向。\n"
                f"命令: {last_result.get('command','')} 退出码: {last_result.get('exit_code')}\n"
                f"STDOUT: {_short(last_result.get('stdout'))}\n"
                f"STDERR: {_short(last_result.get('stderr'))}"
            )
            suggestion = llm_completion(prompt, temperature=0.2, max_tokens=80).strip()
        except Exception:
            suggestion = "失败，建议检查命令、权限、路径或网络。"
        return False, True, suggestion
    if current_index >= steps_len:
        return True, False, "所有步骤执行成功"
    return False, False, f"将执行索引 {current_index}"


