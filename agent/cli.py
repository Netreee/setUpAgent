#!/usr/bin/env python3
"""
独立任务执行器 CLI（不修改 main.py）。

用法示例：
  python -m agent.cli --goal "安装requirements中所有的依赖"
"""

import argparse
from typing import Dict, Any
from agent.workflow import create_task_graph
from agent.task_types import AgentState


def run(goal: str, recursion_limit: int = 100):
    agent = create_task_graph()

    initial_state: AgentState = {
        "messages": [{"role": "user", "content": goal}],
        "observation": "",
        "is_complete": False,
        "failed": False,
    }

    step = 1
    def _extract_last_tool_call_args(messages):
        """从消息中提取最近一次 run_instruction 的工具调用入参。"""
        try:
            for msg in reversed(messages or []):
                tool_calls = getattr(msg, "tool_calls", None)
                if not tool_calls:
                    continue
                for call in reversed(tool_calls):
                    name = call.get("name") if isinstance(call, dict) else None
                    if name == "run_instruction":
                        args = call.get("args", {})
                        return {
                            "nl_instruction": args.get("nl_instruction"),
                            "timeout": args.get("timeout"),
                        }
        except Exception:
            pass
        return {}

    for event in agent.stream(initial_state, config={"recursion_limit": recursion_limit}):
        for node_name, node_state in event.items():
            pass
        step += 1


def main():
    parser = argparse.ArgumentParser(description="独立任务执行器（线性计划→顺序执行→观察）")
    parser.add_argument("--goal", required=True, help="用户任务描述，例如：安装requirements中所有的依赖")
    parser.add_argument("--recursion-limit", type=int, default=100, help="LangGraph 最大递归层数（默认100）")
    args = parser.parse_args()
    run(args.goal, recursion_limit=args.recursion_limit)


if __name__ == "__main__":
    main()


