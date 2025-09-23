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
    prompt = (
        "你是观察者。基于结果、模式与事实，决定下一跳路由与可能的事实增量。\n"
        f"模式: {mode} 周期: {episode}\n"
        f"目标: {task.get('goal','')}\n"
        f"计划步骤标题序列: {titles}\n"
        f"当前索引: {current_index}\n"
        f"最近一次结果: {_json.dumps(last_result or {}, ensure_ascii=False)[:1200]}\n"
        f"已知事实: {_json.dumps(facts or {}, ensure_ascii=False)[:1200]}\n\n"
        "路由集合: decide | repeat_step | skip_step | plan | switch_mode | end\n"
        "如需切换模式，请给出新的模式 discover/execute。可选给出 facts_delta 与 insert_steps。\n\n"
        "仅输出 JSON：{\n"
        "  \"route\": \"decide|repeat_step|skip_step|plan|switch_mode|end\",\n"
        "  \"mode\": \"discover|execute\" | null,\n"
        "  \"facts_delta\": { } | null,\n"
        "  \"notes\": \"一句话原因\",\n"
        "  \"insert_steps\": [{\"title\":\"...\",\"instruction\":\"...\"}] | null\n"
        "}"
    )
    debug.note("observer_prompt", prompt)
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
    return data


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


