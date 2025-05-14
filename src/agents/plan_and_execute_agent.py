import logging
import time
from typing import Any, Dict, List, Optional

from litellm import completion

from src.agents.base import Agent
from src.envs.base import Env
from src.types import AgentRunResult
from src.utils import convert_message_to_action

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
# Suppress LiteLLM logs specifically
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

PLANNER_INSTRUCTION = """
You are a planning agent. Given a user request, break it down into a numbered list of clear, concrete steps that, if executed in order, will fully accomplish the user's goal. If the request is already simple, you may return a single step. Do not execute any steps yourself.
If there exists an observeration about previous query attempts, reflect on it to improve the plan.

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

        logging.info("Starting PlanAndExecuteAgent run for task_index=%s", task_index)

        # Step 1: Planning
        planner_messages = [
            {
                "role": "system",
                "content": PLANNER_INSTRUCTION,
            },
            {"role": "user", "content": obs_user},
        ]
        for _ in range(max_num_steps):
            logging.info("Planning: requesting plan for user request: %s", obs_user)
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

            num_tries = 0
            while True:
                try:
                    res = completion(
                        messages=planner_messages,
                        model=self.model,
                        temperature=self.temperature,
                    )
                    agent_cost += res._hidden_params["response_cost"]
                    if res.choices[0].message.content is None:
                        if num_tries >= 3:
                            print("Max retries reached for planner, breaking...")
                            break
                        print("Empty response from planner, retrying...")
                        num_tries += 1
                        time.sleep(10)
                        continue
                    break
                except Exception as e:
                    time.sleep(10)
                    print(e, end="\r")
            plan_text = res.choices[0].message.content
            if plan_text is None:  # Handle empty response
                logging.error("Received empty response from planner.")
                break
            logging.info("Received plan:\n%s", plan_text)
            plan = [
                line.strip(" .")
                for line in plan_text.split("\n")
                if line.strip()
                and (line.strip()[0].isdigit() or len(plan_text.split("\n")) == 1)
            ]
            if not plan:
                plan = [plan_text.strip()]
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
                    logging.info(
                        "Executing step %d/%d: %s", cur_step + 1, len(plan), step
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
                env_response = env.step(action)
                logging.info("Step result: %s", env_response.observation)
                reward = env_response.reward
                env_info = {**env_info, **env_response.info.model_dump()}
                plan_executed.append(step)
                if action.name != "respond":
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
                    messages.extend(
                        [
                            next_message,
                            {"role": "user", "content": env_response.observation},
                        ]
                    )
                    logging.info("User response generated, breaking execution loop.")
                    is_user_response = True
                    latest_plan = step
                    latest_result = plan_results[-1]
                    break

            if env_response.done:
                logging.info("Environment signaled done. Exiting.")
                done = True
                break

            if is_user_response:
                # update
                plan_executed = []
                plan_results = []
                obs_user = env_response.observation
                logging.info(
                    "Switching back to planner with new user observation: %s", obs_user
                )

                # go back to planner
                continue

            # Step 3: Replanning or Respond
            if not done:
                logging.info("Replanning: steps executed: %s", plan_executed)
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
                logging.info("Replanner reply: %s", replanner_reply)
                planner_messages.append(
                    {"role": "assistant", "content": replanner_reply}
                )
            else:
                messages.extend(executor_messages)
        logging.info(
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
