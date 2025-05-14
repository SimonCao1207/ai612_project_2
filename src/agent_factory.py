from typing import Any, Dict, List

from src.agents.base import Agent


def get_agent(
    tools_info: List[Dict[str, Any]],
    model: str,
    agent_strategy: str = "tool-calling",
    temperature: float = 0.0,
    rule: str = "",
) -> Agent:
    if agent_strategy == "tool-calling":
        from src.agents.tool_calling_agent import ToolCallingAgent

        return ToolCallingAgent(
            tools_info=tools_info,
            model=model,
            temperature=temperature,
            rule=rule,
        )
    elif agent_strategy == "plan-and-execute":
        from src.agents.plan_and_execute_agent import PlanAndExecuteAgent

        return PlanAndExecuteAgent(
            tools_info=tools_info,
            model=model,
            temperature=temperature,
            rule=rule,
        )
