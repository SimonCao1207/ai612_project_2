import time
from typing import Any, Dict, List, Optional

from litellm import completion

from src.agents.base import Agent
from src.envs.base import Env
from src.log import Logger
from src.types import AgentRunResult
from src.utils import convert_message_to_action

logger = Logger()

PLANNER_INSTRUCTION = """
You are a planning agent. Given a user request, break it down into a numbered list of clear, concrete steps that, if executed in order, will fully accomplish the user's goal. If the request is already simple, you may return a single step. Do not execute any steps yourself.
If there exists an observeration about previous query attempts, reflect on it to improve the plan.
If user's request is to reconfirm the previous query, you should not generate a new plan but just return an empty string.

You have access to the following tools:
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

    def _get_plan_from_planner(self, planner_messages):
        while True:
            try:
                res = completion(
                    messages=planner_messages,
                    model=self.model,
                    temperature=self.temperature,
                )
                plan_text = res.choices[0].message.content
                agent_cost = res._hidden_params["response_cost"]
                if plan_text is not None:
                    plan = [
                        line.strip(" .")
                        for line in plan_text.split("\n")
                        if line.strip()
                        and (
                            line.strip()[0].isdigit() or len(plan_text.split("\n")) == 1
                        )
                    ]
                else:
                    plan = []
                return plan, plan_text, agent_cost
            except Exception as e:
                time.sleep(10)
                print(e, end="\r")
                return [], None, 0.0

    def _execute_action_and_update_env(self, env, action, env_info):
        """
        Executes the given action in the environment, updates reward and env_info, and returns env_response, reward, env_info.
        """
        env_response = env.step(action)
        reward = env_response.reward
        env_info = {**env_info, **env_response.info.model_dump()}
        return env_response, reward, env_info

    def run(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> AgentRunResult:
        agent_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs_user = env_reset_res.observation
        env_info = env_reset_res.info.model_dump()
        reward = 0.0
        plan = []
        plan_executed = []
        latest_plan = None
        latest_result = None
        plan_results = []
        done = False
        is_user_response = False

        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": obs_user},
        ]

        logger.log("Starting PlanAndExecuteAgent run for task_index=%s", task_index)

        # Step 1: Planning
        planner_messages = [
            {
                "role": "system",
                "content": PLANNER_INSTRUCTION,
            },
            {"role": "user", "content": obs_user},
        ]
        for _ in range(max_num_steps):
            logger.log_chat(obs_user, "User request")
            if (
                latest_plan is not None
                and latest_result is not None
                and is_user_response
            ):
                planner_messages.append(
                    {
                        "role": "user",
                        "content": f"User request {obs_user}\nLatest plan executed: {latest_plan}\nResults so far: {latest_result}",
                    }
                )

            plan, plan_text, planner_cost = self._get_plan_from_planner(
                planner_messages
            )
            agent_cost += planner_cost
            if len(plan) == 0:
                logger.log(level="error", message="No Plan generated.")
                i_am_sure_bitch = "Yes, I am sure with my previous response."
                action = convert_message_to_action(
                    {
                        "role": "assistant",
                        "content": i_am_sure_bitch,
                    }
                )
                logger.log_chat(i_am_sure_bitch, "Agent Response")

                env_response, reward, env_info = self._execute_action_and_update_env(
                    env, action, env_info
                )

                if env_response.done:
                    logger.log_chat(env_response.content, "User response")
                    logger.log("Environment signaled done. Exiting.")
                    done = True
                    break
                else:
                    continue
            else:
                logger.log_code("Plan", "\n".join(plan))

            # Step 2: Execute each step
            executor_messages = [
                {
                    "role": "system",
                    "content": EXECUTOR_INSTRUCTION + "\nRules:\n" + self.rule,
                },
            ]

            is_user_response = False
            for cur_step in range(len(plan) + 1):
                if cur_step < len(plan):
                    step = plan[cur_step]
                    logger.log_rule(
                        f"Executing step {cur_step + 1}/{len(plan)}: {step}"
                    )
                    executor_messages.append(
                        {
                            "role": "user",
                            "content": f"User request: {obs_user}\nCurrent step: {step} \nLatest step executed: {latest_plan}\nResults so far: {latest_result}",
                        }
                    )
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
                        time.sleep(10)
                        print(e, end="\r")
                next_message = res.choices[0].message.model_dump()
                action = convert_message_to_action(next_message)
                env_response, reward, env_info = self._execute_action_and_update_env(
                    env, action, env_info
                )
                plan_executed.append(step)
                if action.name != "respond":
                    logger.log_chat(
                        next_message["tool_calls"][0]["function"]["arguments"],
                        f"Tool Call : {action.name}",
                    )
                    next_message["tool_calls"] = next_message["tool_calls"][:1]
                    new_messages = [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                    executor_messages.extend(new_messages)
                    messages.extend(new_messages)
                    plan_results.append(env_response.observation)
                else:
                    logger.log_chat(next_message["content"], "Agent Response")
                    messages.extend(
                        [
                            next_message,
                            {"role": "user", "content": env_response.observation},
                        ]
                    )
                    logger.log("User response generated, breaking execution loop.")
                    is_user_response = True
                    latest_plan = step
                    latest_result = plan_results[-1]
                    break

            if env_response.done:
                logger.log("Environment signaled done. Exiting.")
                done = True
                break

            if is_user_response:
                plan_executed = []
                plan_results = []
                obs_user = env_response.observation
                logger.log("Switching back to planner with new user observation:")
                continue

            # Step 3: Replanning or Respond
            if not done:
                logger.log_code(
                    "Replanning: steps executed: %s", "\n".join(plan_executed)
                )
                replanner_messages = [
                    {
                        "role": "system",
                        "content": REPLANNER_INSTRUCTION + "\nRules:\n" + self.rule,
                    },
                    {
                        "role": "user",
                        "content": f"User request: {obs_user} \nSteps executed: {plan_executed}\nResults so far: {plan_results}",
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
                        time.sleep(10)
                        print(e, end="\r")
                replanner_reply = res.choices[0].message.content
                logger.log_chat(replanner_reply, "Replanner reply")
                planner_messages.append(
                    {"role": "assistant", "content": replanner_reply}
                )
            else:
                messages.extend(executor_messages)
        logger.log(
            "Run complete. Final reward: %s, agent_cost: %s",
            reward,
            round(agent_cost, 8),
        )
        return AgentRunResult(
            reward=reward,
            messages=messages,
            agent_cost=round(agent_cost, 8),
            info=env_info,
        )
