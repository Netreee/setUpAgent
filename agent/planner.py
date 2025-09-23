from typing import Tuple, Optional, Dict, Any, List
from uuid import uuid4
from string import Template
from config import get_config
from utils import llm_completion
from agent.task_types import Task, TaskStep
from agent.debug import dispInfo, debug

PLANNER_PROMPT_TEMPLATE = Template(
    """
你是一个任务规划器。根据用户的任务与“当前周期模式”生成该周期的线性步骤。

【输出格式】
- 仅输出JSON，不要任何其他说明、解释或多余文字。
- JSON结构：{"title": "本周期标题", "steps": [{"title": "步骤标题", "instruction": "自然语言指令"}, ...]}

【环境与模式】
- 工作目录：$agent_work_root
- 操作系统：Windows；Shell：PowerShell；指令将被自动翻译为单条 PowerShell 命令
- 当前模式：$mode  （discover|execute）
- 周期序号：$episode
- 已知事实 facts（供参考）：$facts_json

【步骤建议（软约束）】
- discover：优先做只读与低风险操作以收集上下文（列目录、读取小文件片段、检测版本/配置、解析依赖元数据等）。
- execute：基于已知事实推进目标（安装/构建/运行/验证等）。
- 步骤为“做什么”的自然语言；每步对应单条命令；必要时明确相对路径。

【增量重规划】
- 若提供了上下文（已完成/剩余/最近结果），请进行“增量重规划”：
  1) 保留已完成步骤；2) 仅生成“剩余步骤”（可在开头插入修补步骤）；
  3) 保证线性可执行，每步单条命令；4) 输出 JSON 中 steps 代表“剩余步骤”。

【上下文信息】
- 是否提供上下文：$has_context
- 已完成步骤（标题序列）：$completed_titles
- 既有剩余步骤（标题序列）：$remaining_titles
- 最近一次失败/结果摘要：exit_code=$last_exit, cmd=$last_cmd
- STDOUT: $last_stdout
- STDERR: $last_stderr

【示例（可借鉴表达方式）】
- "发现当前仓库的关键文件并输出JSON摘要"
- "在当前目录创建一个名为test.txt的空文件"
- "安装项目为可编辑模式"
- "安装requirements.txt中的所有Python依赖包"
- "将当前目录下的所有.py文件复制到backup文件夹"
- "使用pip安装requests和numpy包"
- "创建一个名为output的文件夹，并在其中创建README.md文件"

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
    mode = (context or {}).get("mode", "discover")
    episode = (context or {}).get("episode", 1)
    facts = (context or {}).get("facts", {})
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
        facts_json=_json.dumps(facts, ensure_ascii=False)[:1500],
        has_context="是" if context else "否",
        completed_titles=_titles(completed_steps),
        remaining_titles=_titles(remaining_steps),
        last_exit=str(last_result.get("exit_code", "")),
        last_cmd=str(last_result.get("command", ""))[:200],
        last_stdout=str(last_result.get("stdout", ""))[:300],
        last_stderr=str(last_result.get("stderr", ""))[:300],
    )

    debug.note("planner_prompt", user_prompt)
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


