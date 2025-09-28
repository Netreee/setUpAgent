from typing import TypedDict, List, Optional


class TaskStep(TypedDict, total=False):
    """
    任务步骤定义。

    字段：
    - id: 步骤序号（从1开始）
    - title: 步骤标题，简要说明该步骤要做什么
    - instruction: 步骤的自然语言指令，将交给 run_single 执行
    - timeout: 可选，单步超时时间（秒），未提供则用默认值
    """
    id: int
    title: str
    instruction: str
    timeout: Optional[int]


class Task(TypedDict):
    """
    任务定义。

    字段：
    - id: 任务ID（字符串）
    - goal: 用户的任务描述
    - steps: 步骤列表
    """
    id: str
    goal: str
    steps: List[TaskStep]


class StepResult(TypedDict, total=False):
    """
    单步执行结果（run_single 返回的结构）。

    字段：
    - step_id: 对应步骤的 id（可选，由编排层在需要时注入）
    - exit_code: 退出码
    - stdout: 标准输出
    - stderr: 标准错误
    - command: 实际执行的命令
    - work_dir: 执行时的工作目录
    """
    step_id: Optional[int]
    exit_code: int
    stdout: str
    stderr: str
    command: str
    work_dir: str


class AgentState(TypedDict, total=False):
    """
    任务代理的图状态。

    字段：
    - messages: 对话历史（留作扩展）
    - task: 当前任务
    - plan_text: 人类可读的计划文本（给日志/调试用）
    - current_step_index: 当前准备执行的步骤索引（0-based）
    - last_result: 上一次步骤执行结果
    - is_complete: 是否完成（成功或失败均视为完成）
    - failed: 是否失败
    - observation: 观察结论文字
    """
    messages: List[dict]
    task: Task
    plan_text: str
    current_step_index: int
    last_result: Optional[StepResult]
    is_complete: bool
    failed: bool
    observation: str
    # 以下为扩展字段：用于路由与重规划
    replan_requested: bool
    route: str
    # 会话ID：用于持久 Shell 会话
    session_id: Optional[str]
    # 多周期/多模式
    mode: str  # "discover" | "execute"
    episode: int
    facts: dict
    # 从 README 中提取的结构化信息
    READMEinfo: dict
    # 由 observer 维护：已完成步骤的标题集合（用于 planner 提示防止重复）
    finished_titles: list[str]


