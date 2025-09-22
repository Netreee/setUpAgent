#!/usr/bin/env.example python3
"""
LangGraph Plan-Execute-Observe Agent 模板
这是一个MVP实现，用户可以在这里填充具体的业务逻辑。
"""

from typing import TypedDict, List, Any, Dict
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from config import get_config, get_llm_config


# 定义状态模型
class AgentState(TypedDict):
    """Agent的状态定义"""
    messages: List[Dict[str, str]]  # 对话历史
    plan: str  # 当前计划
    execution_result: Any  # 执行结果
    observation: str  # 观察结果
    is_complete: bool  # 是否完成


# 节点函数 - Plan阶段
def plan_node(state: AgentState) -> AgentState:
    """
    规划阶段：分析当前状态，制定执行计划
    用户在这里填充具体的规划逻辑
    """
    # 获取最新用户消息
    latest_message = state["messages"][-1]["content"] if state["messages"] else ""

    # TODO: 用户填充具体的规划逻辑
    # 例如：调用LLM分析用户需求，制定执行计划
    plan = f"根据用户输入 '{latest_message}' 制定的执行计划"

    return {
        **state,
        "plan": plan,
        "is_complete": False  # 继续执行循环
    }


# 节点函数 - Execute阶段
def execute_node(state: AgentState) -> AgentState:
    """
    执行阶段：根据计划执行具体操作
    用户在这里填充具体的执行逻辑
    """
    plan = state["plan"]

    # TODO: 用户填充具体的执行逻辑
    # 例如：调用API、执行计算、处理数据等
    execution_result = f"执行计划 '{plan}' 的结果"

    return {
        **state,
        "execution_result": execution_result
    }


# 节点函数 - Observe阶段
def observe_node(state: AgentState) -> AgentState:
    """
    观察阶段：分析执行结果，决定是否继续或结束
    用户在这里填充具体的观察逻辑
    """
    execution_result = state["execution_result"]

    # TODO: 用户填充具体的观察逻辑
    # 例如：检查结果是否符合预期，决定是否需要继续
    observation = f"观察执行结果：{execution_result}"

    # 使用配置中的完成条件判断
    # TODO: 用户根据业务逻辑定义完成条件
    config = get_config()
    # 简单的演示条件：消息数量超过配置的阈值
    is_complete = len(state["messages"]) > config.completion_threshold

    return {
        **state,
        "observation": observation,
        "is_complete": is_complete
    }


# 构建工作流图
def create_agent_graph():
    """创建LangGraph工作流"""

    # 创建状态图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("plan", plan_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("observe", observe_node)

    # 设置起始节点
    workflow.set_entry_point("plan")

    # 添加边 - 从plan到execute
    workflow.add_edge("plan", "execute")

    # 从execute到observe
    workflow.add_edge("execute", "observe")

    # 从observe回到plan（循环）或结束
    workflow.add_conditional_edges(
        "observe",
        # 条件函数：判断是否完成
        lambda state: END if state["is_complete"] else "plan",
        {
            END: END,  # 结束
            "plan": "plan"  # 继续循环
        }
    )

    return workflow.compile()


# 主函数
def main():
    """主函数 - 运行Agent"""

    # 创建Agent
    agent = create_agent_graph()

    # 获取配置
    config = get_config()

    # 初始化状态
    initial_state: AgentState = {
        "messages": [{"role": "user", "content": ""}],
        "plan": "",
        "execution_result": None,
        "observation": "",
        "is_complete": False
    }

    print("=== LangGraph Plan-Execute-Observe Agent ===")
    print(f"项目名称: {config.project_name}")
    print(f"版本: {config.version}")
    print(f"初始用户输入: {initial_state['messages'][0]['content']}")
    if config.debug_mode:
        print(f"调试模式已启用")
        print(f"最大迭代次数: {config.max_iterations}")
        print(f"完成阈值: {config.completion_threshold}")
    print("\n开始执行循环...\n")

    # 运行Agent
    step = 1
    for event in agent.stream(initial_state):
        print(f"=== 第 {step} 轮 ===")

        for node_name, node_state in event.items():
            print(f"节点: {node_name}")
            print(f"计划: {node_state.get('plan', 'N/A')}")
            print(f"执行结果: {node_state.get('execution_result', 'N/A')}")
            print(f"观察结果: {node_state.get('observation', 'N/A')}")
            print(f"是否完成: {node_state.get('is_complete', 'N/A')}")
            print("-" * 50)

        step += 1

        # 限制循环次数，避免无限循环
        if step > config.max_iterations:
            print(f"达到最大循环次数({config.max_iterations})，强制结束")
            break

    print("\n=== 执行完成 ===")


if __name__ == "__main__":
    main()
