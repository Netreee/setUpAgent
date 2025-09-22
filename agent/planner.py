from typing import Tuple, Optional, Dict, Any, List
from uuid import uuid4
from string import Template
from config import get_config
from utils import llm_completion
from agent.task_types import Task, TaskStep
from agent.debug import dispInfo, debug

PLANNER_PROMPT_TEMPLATE = Template(
    """
你是一个任务规划器。根据用户的任务描述，产出一个线性的步骤列表。要求：

【输出格式】
- 仅输出JSON，不要任何其他说明、解释或多余文字。
- JSON结构必须严格遵循：{"title": "总体标题", "steps": [{"title": "步骤标题", "instruction": "具体指令"}, ...]}

【工作环境】
- 当前工作目录：$agent_work_root
- 操作系统：Windows
- 默认Shell：PowerShell
- 所有命令必须在Windows/PowerShell环境中可直接执行

【步骤要求】
- 每个步骤必须能通过单条命令完成
- 指令应详细描述"做什么"，而不是"怎么做"（系统会自动转换为具体命令）
- 指令应考虑上下文和依赖关系
- 如果需要创建文件/目录，请明确路径（相对当前工作目录）
- 如果需要安装依赖，请指定具体包管理器（pip, conda等）

【重规划规则（增量）】
- 若提供了上下文（已完成步骤/剩余步骤/失败信息），请进行“增量重规划”而非完整重写：
  1) 保留已完成步骤不变；
  2) 仅生成“剩余步骤”，可在开头插入必要的修补步骤（如环境准备/错误修复）；
  3) 确保新的剩余步骤线性可执行，且每步可由单条命令完成；
  4) 输出的 JSON 中 steps 字段仅表示“剩余步骤”。

【上下文信息】
- 是否提供上下文：$has_context
- 已完成步骤（标题序列）：$completed_titles
- 既有剩余步骤（标题序列）：$remaining_titles
- 最近一次失败/结果摘要：exit_code=$last_exit, cmd=$last_cmd
- STDOUT: $last_stdout
- STDERR: $last_stderr

【指令示例】
以下是良好的指令示例：

正确示例：
- "在当前目录创建一个名为test.txt的空文件"
- "安装requirements.txt中的所有Python依赖包"
- "将当前目录下的所有.py文件复制到backup文件夹"
- "使用pip安装requests和numpy包"
- "创建一个名为output的文件夹，并在其中创建README.md文件"

错误示例（避免这些）：
- "运行命令：mkdir test" （不要包含具体命令语法）
- "使用命令行工具" （过于模糊）
- "完成这个复杂的多步骤任务" （没有具体可执行的指令）

【任务要求】
请为以下任务生成步骤计划：
任务：$goal

请严格按照JSON格式输出，不要添加任何解释文字。
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
    if context:
        completed_steps = list(context.get("completed_steps", []))
        remaining_steps = list(context.get("remaining_steps", []))
        last_result = dict(context.get("last_result", {}))

    def _titles(steps: List[TaskStep]) -> str:
        return ", ".join([s.get("title", f"步骤{i+1}") for i, s in enumerate(steps)]) or "(空)"

    # 使用模板渲染动态内容
    user_prompt = PLANNER_PROMPT_TEMPLATE.safe_substitute(
        agent_work_root=config.agent_work_root,
        goal=goal,
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


