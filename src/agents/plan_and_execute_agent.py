import time
from typing import Any, Dict, List, Optional

from litellm import completion

from src.agents.base import Agent
from src.envs.base import Env
from src.types import AgentRunResult
from src.utils import convert_message_to_action

PLANNER_INSTRUCTION = """
You are a planning agent. Given a user request, break it down into a numbered list of clear, concrete steps that, if executed in order, will fully accomplish the user's goal. If the request is already simple, you may return a single step. Do not execute any steps yourself.
If there exists an observeration about previous query attempts, reflect on it to improve the plan.

You have access to the following tools:
- search_doc: Search for documentation or descriptions about the database tables.
- sql_db_list_tables: List all table names in the database.
- sql_db_query: Execute a SQL query against the database and get the result.
- sql_db_schema: Get the schema and sample rows for specific tables.
- value_substring_search: Retrieve up to k values from a column that contains a specified substring.

When planning, consider which tool(s) would be most helpful for each step.
"""

EXECUTOR_INSTRUCTION = """
You are an execution agent. Given the user's request and a specific step from a plan, execute that step as best as possible using the available tools. If you need more information, use the tools to gather it. Do not skip steps or invent information. Only execute the current step.
"""

EXECUTOR_REMINDER = """
If the current step is about analyzing the query result, then use no tool but only give your best interpretation of the result. 
"""

REPLANNER_INSTRUCTION = """
You are a replanning agent. Given the user's request, the steps already executed, and the results so far, decide if the task is complete. If not, generate a new plan for the remaining work. If complete, respond to the user with the final answer.
"""


class PlanAndExecuteAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        rule: str,
        model: str,
        temperature: float = 0.0,
    ):
        self.tools_info = tools_info
        self.rule = rule
        self.model = model
        self.temperature = temperature

    def run(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> AgentRunResult:
        agent_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs_user = env_reset_res.observation
        env_info = env_reset_res.info.model_dump()
        reward = 0.0
        messages: List[Dict[str, Any]] = []
        plan = []
        plan_executed = []
        plan_results = []
        done = False
        # Step 1: Planning
        planner_messages = [
            {
                "role": "system",
                "content": PLANNER_INSTRUCTION,
            },
            {"role": "user", "content": obs_user},
        ]
        for _ in range(max_num_steps):
            while True:
                try:
                    res = completion(
                        messages=planner_messages,
                        model=self.model,
                        temperature=self.temperature,
                    )
                    agent_cost += res._hidden_params["response_cost"]
                    break
                except Exception as e:
                    time.sleep(3)
                    print(e, end="\r")
            plan_text = res.choices[0].message.content
            plan = [
                line.strip(" .")
                for line in plan_text.split("\n")
                if line.strip()
                and (line.strip()[0].isdigit() or len(plan_text.split("\n")) == 1)
            ]
            if not plan:
                plan = [plan_text.strip()]
            # Step 2: Execute each step
            for step in plan:
                executor_messages = [
                    {
                        "role": "system",
                        "content": EXECUTOR_INSTRUCTION + "\nRules:\n" + self.rule,
                    },
                    {
                        "role": "user",
                        "content": f"User request: {obs_user}\nCurrent step: {step} \nSteps executed: {plan_executed}\nResults so far: {plan_results}",
                    },
                ]
                while True:
                    try:
                        res = completion(
                            messages=executor_messages,
                            model=self.model,
                            tools=self.tools_info,
                            temperature=self.temperature,
                        )
                        agent_cost += res._hidden_params["response_cost"]
                        break
                    except Exception as e:
                        time.sleep(5)
                        print(e, end="\r")
                next_message = res.choices[0].message.model_dump()
                action = convert_message_to_action(next_message)
                env_response = env.step(action)
                reward = env_response.reward
                env_info = {**env_info, **env_response.info.model_dump()}
                plan_executed.append(step)
                plan_results.append(env_response.observation)
                if action.name != "respond":
                    next_message["tool_calls"] = next_message["tool_calls"][:1]
                    executor_messages.extend(
                        [
                            next_message,
                            {
                                "role": "tool",
                                "tool_call_id": next_message["tool_calls"][0]["id"],
                                "name": next_message["tool_calls"][0]["function"][
                                    "name"
                                ],
                                "content": env_response.observation,
                            },
                        ]
                    )
                else:
                    executor_messages.extend(
                        [
                            next_message,
                            {"role": "user", "content": env_response.observation},
                        ]
                    )
                if env_response.done:
                    done = True
                    break
            # Step 3: Replanning or Respond
            if not done:
                replanner_messages = [
                    {
                        "role": "system",
                        "content": REPLANNER_INSTRUCTION + "\nRules:\n" + self.rule,
                    },
                    {
                        "role": "user",
                        "content": f"User request: {obs_user}\nSteps executed: {plan_executed}\nResults so far: {plan_results}",
                    },
                ]
                while True:
                    try:
                        res = completion(
                            messages=replanner_messages,
                            model=self.model,
                            temperature=self.temperature,
                        )
                        agent_cost += res._hidden_params["response_cost"]
                        break
                    except Exception as e:
                        time.sleep(5)
                        print(e, end="\r")
                replanner_reply = res.choices[0].message.content
                messages.extend(replanner_messages)
                messages.append({"role": "assistant", "content": replanner_reply})
                planner_messages.append(
                    {"role": "assistant", "content": replanner_reply}
                )
            else:
                messages.extend(executor_messages)
        return AgentRunResult(
            reward=reward,
            messages=messages,
            agent_cost=round(agent_cost, 8),
            info=env_info,
        )
