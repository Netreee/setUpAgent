from typing import Tuple, Optional, Dict, Any, List
from uuid import uuid4
from string import Template
from config import get_config
from utils import llm_completion
from agent.task_types import Task, TaskStep
from agent.debug import dispInfo, debug

PLANNER_PROMPT_TEMPLATE = Template(
    """
你是一个严格的任务规划器。你的职责是基于“用户任务 + 当前周期模式 + 已知事实（facts）”生成本周期可直接执行的线性步骤。

【仅输出】
- 仅输出 JSON，不得包含解释性文字、Markdown、代码块标记或多余内容。

【输出Schema】
{
  "title": "本周期标题",
  "environment_selection": {
    "installer": "uv|pip|poetry|pdm|conda|pipenv|custom|none",
    "reason": "为何选择该安装/运行通道的简要理由（必须引用事实）",
    "evidence_fact_keys": ["facts中用到的关键键路径，如 files.pyproject.exists" ]
  },
  "steps": [ { "title": "步骤标题", "instruction": "单条可执行命令或自然语言指令" } , ... ],
  "self_check": { "ok": true, "violations": ["若发现与约束/事实冲突，逐条描述；若无冲突则为空数组"] }
}

【硬性约束（必须遵守）】
- facts 具有最高优先级：不得生成与 facts 明确冲突的工具、文件或路径（例如 facts 指示 X 文件不存在，就不得引用它）。
- 当 facts 中已经存在关键路径或状态时，必须据此裁剪步骤：
  * 若 facts.project_root 已存在：禁止输出任何“克隆仓库”类步骤；应直接在该目录继续后续步骤。
  * 若 facts.has_pyproject=true（或 has_setup_py/has_requirements_txt=true）：禁止重复探测是否存在，应直接使用这些文件的信息做决策。
  * 若 facts.readme_read=true：除非需要补充信息，避免再次读取 README。
- 每个步骤的 instruction 字段必须是**纯意图描述**，严禁包含：
  * PowerShell 命令（如 Get-ChildItem、Test-Path）
  * 工具名或半格式化调用（如 files_list:、pyenv_python_info:）
  * 只描述"要完成什么"，不描述"如何完成"（决策层会自动选择工具或生成命令）
- 工作目录：$agent_work_root；操作系统：Windows；Shell：PowerShell。
- 输出中不得重复已完成步骤：以下步骤已经完成：$finished_titles_json。

【instruction 字段规范】
正确示例（纯意图描述）：
  ✓ "列出 setupLab2 目录下的所有文件和文件夹"
  ✓ "检查 transformers 目录是否存在"
  ✓ "读取 transformers/pyproject.toml 文件的内容"
  ✓ "克隆 huggingface/transformers 仓库到工作目录"
  ✓ "使用 pip 在项目根目录安装可编辑模式的包"
  
错误示例（包含命令或工具名）：
  ✗ "Get-ChildItem -Path D:\\setupLab2"           ← 不要输出具体命令
  ✗ "files_list: transformers"                    ← 不要输出工具名
  ✗ "Test-Path D:\\setupLab2\\transformers"      ← 不要输出 PowerShell 命令
  ✗ "pip install -e ."                            ← 不要输出命令，应描述意图

【环境与模式】
- 当前模式：$mode  （discover|execute）
- 周期序号：$episode
- 已知事实 facts（必须遵守）：$facts_json

【步骤与选择指南（优先专用工具，run_instruction 仅兜底）】
- discover 模式：生成探测意图（列目录、检查文件是否存在、读取文件内容、解析配置等）
  * 示例："列出 repo_root 下的所有文件"、"检查 pyproject.toml 是否存在"
- execute 模式：生成执行意图（安装依赖、运行脚本、构建项目等）
  * 示例："使用 pip 安装所有依赖"、"运行项目的测试套件"
- 先做"环境选择"：在 environment_selection 中给出 installer 的枚举值，并用 evidence_fact_keys 指明你依据了哪些 facts；reason 需简要引用这些 facts。
- 所有步骤都用意图描述，决策层会根据意图自动选择：
  * 专用工具（files_*/pyenv_*）：用于文件系统探测和 Python 环境分析
  * Git 相关操作优先使用结构化 Git 工具（由决策层映射）：
    - git_repo_status：检查/确认仓库状态（是否为仓库、origin、分支）
    - git_ensure_cloned：确保仓库已在工作区内可用（若不存在则浅克隆，避免重复克隆）
  * 通用执行（run_instruction）：仅用于无法由专用工具完成的命令，避免直接使用 shell 级 git clone
- 禁止描述目录切换操作，需要操作不同目录时直接在意图中说明目标路径（引用 facts 中的路径键）。
- 若 facts 不足以直接做出选择，应在 discover 中先安排最少量的"存在性探测"步骤，然后再进行安装/运行步骤。

【Git 操作策略】
- instruction 中不得出现任何 git 命令；保持纯“意图描述”。
- 与仓库相关的意图（如“确认仓库是否存在并可用”“确保仓库在工作目录可用”），由决策层优先映射到 git_repo_status / git_ensure_cloned 工具，而非 shell。
- execute 模式下若 facts.project_root 已存在：禁止产出“克隆”类意图。必要时可产出“检查仓库状态”的意图（决策层→git_repo_status）。
- 示例：
  * "确认仓库是否已存在并可用" → 决策层会使用 git_repo_status
  * "确保仓库已在工作目录可用（若不存在则克隆）" → 决策层会使用 git_ensure_cloned

【execute 模式严格约束与推理过程】
当 $mode == "execute" 时，必须遵守以下规则：
1. **禁止探测性步骤**：不得生成任何探测性步骤（如 files_exists, files_list, files_find）
2. **禁止重复克隆**：若 facts 中已有 project_root，说明仓库已克隆完成，禁止生成 git clone 步骤
3. **基于facts生成步骤**：每个步骤必须引用 facts 中已确认存在的文件/目录/工具
4. **推理过程示例**：
   ```
   检查facts: {"project_root": "D:\\work\\repo", "has_pyproject": true}
   推理: 仓库已存在，有pyproject.toml，可直接安装
   生成步骤: "在 project_root 目录下使用 pip 安装可编辑模式的包"
   ```
5. **Facts不足处理**：若 facts 不足以支撑任何 execute 步骤，应在 self_check.violations 中说明并返回空 steps 数组

【Facts应用检查清单】
在生成每个步骤前，必须检查：
- ✓ 是否需要project_root？检查facts中是否存在
- ✓ 是否需要特定文件？检查facts中对应的has_xxx字段
- ✓ 是否与已有facts冲突？如facts显示文件不存在但步骤要使用该文件
- ✓ 是否忽略了关键facts？如project_root存在但仍要git clone

【discover 模式约束与推理（严格利用 facts，减少无效探测）】
1. **意图描述**：所有步骤用意图描述（不要输出命令），决策层会自动选择工具
2. **避免重复**：严格检查 $finished_titles_json，不得重复已完成的步骤
3. **Facts应用推理**：
   ```
   检查facts: {"project_root": "D:\\work\\repo"}
   推理: 仓库已存在，无需克隆，直接探测内部结构
   生成步骤: "列出 project_root 目录下的所有文件和文件夹"
   ```
4. **智能跳过已知信息**：
   - 若facts中已有project_root，跳过git clone步骤
   - 若facts中已有has_pyproject=true，跳过检查pyproject.toml是否存在
   - 若facts中已有readme_read=true，跳过读取README步骤
5. **意图描述示例**：
   - "列出 X 目录的文件" → 决策层会选择 files_list 工具
   - "检查 X 文件是否存在" → 决策层会选择 files_exists 工具
   - "读取 X 文件的内容" → 决策层会选择 files_read 工具

【增量重规划】
- 若提供了上下文（已完成/剩余/最近结果），请“只生成剩余步骤”：
  1) 保留已完成步骤；2) 仅输出“剩余步骤”（可在开头插入必要的修补步骤）；
  3) 保证线性可执行，每步单条命令；4) 输出 JSON 中 steps 代表“剩余步骤”。

【一致性自检】
- 在输出前进行自我检查：逐条核对"硬性约束"与 facts；如有冲突，先在你这一步完成修正后再输出；self_check.ok 必须为 true。
- **Facts一致性验证**：
  1. 检查每个步骤是否与facts冲突（如facts显示文件不存在，但步骤要读取该文件）
  2. 检查是否忽略了关键facts（如project_root存在但仍要git clone）
  3. 检查是否充分利用了facts（如has_pyproject=true但选择了错误的安装器）
  4. 如发现问题，必须在violations中详细说明并修正步骤

注意：模式切换由上游 observer 直接写入 state.mode，planner 仅按 state.mode 规划，不处理模式切换。

请为以下任务生成该周期步骤：
任务：$goal

仅输出JSON。
"""
)


# @dispInfo("planner")
def plan_with_llm(goal: str, context: Optional[Dict[str, Any]] = None) -> Tuple[Task, str, str]:
    """
    使用 LLM 生成线性计划。

    返回：
    - Task: 结构化任务
    - plan_text: 人类可读的计划摘要
    """
    config = get_config()

    # 处理上下文（用于增量重规划）
    completed_steps: List[TaskStep] = []
    remaining_steps: List[TaskStep] = []
    last_result: Dict[str, Any] = {}
    mode = (context or {}).get("mode", "execute")
    episode = (context or {}).get("episode", 1)
    facts = (context or {}).get("facts", {})
    finished_titles: List[str] = (context or {}).get("finished_titles", []) or []
    if context:
        completed_steps = list(context.get("completed_steps", []))
        remaining_steps = list(context.get("remaining_steps", []))
        last_result = dict(context.get("last_result", {}))

    def _titles(steps: List[TaskStep]) -> str:
        return ", ".join([s.get("title", f"步骤{i+1}") for i, s in enumerate(steps)]) or "(空)"

    # 使用模板渲染动态内容
    import json as _json
    user_prompt = PLANNER_PROMPT_TEMPLATE.safe_substitute(
        agent_work_root=config.agent_work_root,
        goal=goal,
        mode=str(mode),
        episode=str(episode),
        facts_json=_json.dumps(facts, ensure_ascii=False)[:4000],
        has_context="是" if context else "否",
        completed_titles=_titles(completed_steps),
        remaining_titles=_titles(remaining_steps),
        last_exit=str(last_result.get("exit_code", "")),
        last_cmd=str(last_result.get("command", ""))[:200],
        last_stdout=str(last_result.get("stdout", ""))[:300],
        last_stderr=str(last_result.get("stderr", ""))[:300],
        finished_titles_json=_json.dumps(finished_titles, ensure_ascii=False),
    )

    # debug.note("planner_prompt", user_prompt)  # 提示词太长，不记录
    resp = llm_completion(user_prompt, temperature=0.1, max_tokens=1000)
    debug.note("planner_raw_resp", resp)

    # 改进的JSON解析（更强的容错性）
    import json
    import re

    steps: list[TaskStep] = []
    plan_title = goal

    # 尝试多种方式提取JSON
    json_text = resp.strip()

    # 方法1: 直接JSON解析
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        # 方法2: 尝试从文本中提取JSON（处理可能的markdown代码块）
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', json_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                # 方法3: 尝试找到JSON对象边界
                start = json_text.find('{')
                end = json_text.rfind('}') + 1
                if start != -1 and end > start:
                    try:
                        data = json.loads(json_text[start:end])
                    except json.JSONDecodeError:
                        data = None
                else:
                    data = None
        else:
            data = None

    # 解析任务数据
    if data and isinstance(data, dict):
        plan_title = data.get("title", goal)
        raw_steps = data.get("steps", [])

        if isinstance(raw_steps, list):
            for idx, s in enumerate(raw_steps, start=1):
                if isinstance(s, dict):
                    title = s.get("title", "").strip()
                    instruction = s.get("instruction", "").strip()

                    if not title:
                        title = f"步骤{idx}"
                    if not instruction:
                        continue

                    steps.append({
                        "id": idx,
                        "title": title,
                        "instruction": instruction,
                    })

    # 如果解析失败或步骤为空，退化为单步骤
    if not steps:
        steps = [{"id": 1, "title": "执行任务", "instruction": goal}]

    # 合并“已完成步骤 + 新的剩余步骤”（增量重规划）
    final_steps: List[TaskStep] = []
    if completed_steps:
        final_steps.extend(completed_steps)
    base = len(final_steps)
    for i, s in enumerate(steps, start=1):
        final_steps.append({
            "id": base + i,
            "title": s.get("title", f"步骤{base+i}"),
            "instruction": s.get("instruction", ""),
            **({"timeout": s["timeout"]} if "timeout" in s else {}),
        })

    task: Task = {
        "id": uuid4().hex[:8],
        "goal": goal,
        "steps": final_steps,
    }
    debug.note("final_steps", final_steps)

    plan_text = f"任务: {plan_title}\n共 {len(final_steps)} 个步骤。"
    if len(final_steps) > 1:
        step_titles = [s["title"] for s in final_steps]
        plan_text += f"\n步骤列表: {' → '.join(step_titles)}"

    return task, plan_text, resp


