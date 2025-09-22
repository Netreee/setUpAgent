from typing import Tuple
from agent.task_types import Task, StepResult
from utils import llm_completion
from agent.debug import dispInfo, debug


# @dispInfo("observer")
def observe(task: Task, current_index: int, last_result: StepResult | None) -> Tuple[bool, bool, str]:
    """
    观察器：根据最后一次执行结果与当前位置，决定是否完成或失败。

    返回：(is_complete, failed, observation)
    - is_complete: 是否已完成（成功走到终点或中途失败都视为完成当前回合）
    - failed: 是否失败
    - observation: 文本描述
    """
    steps_len = len(task.get("steps", []))

    if steps_len == 0:
        return True, False, "没有可执行的步骤，视为完成"

    if last_result is None and current_index == 0:
        return False, False, "尚未开始执行"

    # 失败：最近一次结果非0（引入LLM分析，给出纠错建议简述）
    if last_result is not None and last_result.get("exit_code", -1) != 0:
        try:
            prompt = (
                "你是一个软件自动化执行的观察者。\n"
                "请分析一次命令执行失败的原因，并给出简短的纠错建议（不超过140字）。\n"
                f"任务目标: {task.get('goal','')}\n"
                f"步骤ID: {last_result.get('step_id')} 当前索引: {current_index}\n"
                f"命令: {last_result.get('command','')}\n"
                f"退出码: {last_result.get('exit_code')}\n"
                f"STDOUT: {last_result.get('stdout','')[:500]}\n"
                f"STDERR: {last_result.get('stderr','')[:500]}\n"
                "仅输出纠错建议一句话。"
            )
            debug.note("observer_prompt", prompt)
            suggestion = llm_completion(prompt, temperature=0.2, max_tokens=120).strip()
            debug.note("observer_raw_resp", suggestion)
        except Exception:
            suggestion = "执行失败。建议检查命令语法、权限、路径或网络。"
        sid = last_result.get("step_id") if isinstance(last_result, dict) else None
        sid_text = str(sid) if sid is not None else f"索引{current_index}"
        # 失败但不终止，以便进入重规划
        return False, True, f"步骤 {sid_text} 失败：{suggestion}"

    # 成功走到末尾
    if current_index >= steps_len:
        return True, False, "所有步骤执行成功"

    # 仍需继续（对齐推进语义，提示下一个将要执行的索引）
    return False, False, f"已完成步骤索引 {current_index - 1 if current_index > 0 else -1}，将执行索引 {current_index}"


