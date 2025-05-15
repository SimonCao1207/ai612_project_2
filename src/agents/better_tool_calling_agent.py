import logging
import time
from typing import Any, Dict, List, Optional

from litellm import completion

from src.agents.base import Agent
from src.envs.base import Env
from src.log import Logger
from src.types import AgentRunResult
from src.utils import convert_message_to_action

logger = Logger()

# Suppress LiteLLM logs specifically
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


TOOL_CALLING_INSTRUCTION = """- You are a SQL agent that translates natural language questions into precise SQL queries for electronic health records (EHR).
- You are currently engaged in a conversation with a user who wants to retrieve data from an EHR database.
- Your first question to the user should be to ask for the specific information they want to retrieve without breaking down its goal into smaller sub-questions. 
- Force the user to provide a detail if/else statement in the goal provided. For example, "Can you tell me what is the if/else statement in your goal?".
- If the user's request is ambiguous or missing crucial information (e.g., filtering criteria), you must ask clarifying questions in plain language.
- If the user ask to reconfirm or to asking question that you think you have already answered, you should repeat the previous answer to the user.
- You can interact with the database to learn more about its schema or the values stored in it by using the tools provided.
- Do not invent or fabricate any information not provided by the user or the tools.
- You should make at most one tool call at a time.
- If you do call a tool, do not respond to the user in that same turn.
- Do not generate SQL queries directly without knowing the database schema and values intended to be used in the SQL query by calling substring_search_tool.
- When the user asks for specific diagnoses, procedures, medications, or lab tests, try your best to use the tool to search for relevant information in the database and determine if it relates to the user's request.
- Only when you have gathered all necessary information from the user or the database, produce a single, valid SQL query that fully reflects the user's request.
- Avoid partial or speculative queries that could cause confusion or yield inaccurate results.
- Your performance is evaluated based on the latest SQL query you generate, so when generating a new SQL query for the user's request, avoid relying on previous results but instead rewrite it from scratch to fully capture the user's intent and ensure it is accurately assessed.
"""

DECOMPOSITION_INSTRUCTION = """ You are a agent that breaks down complex tasks from user request into smaller, manageable subtasks.
For example: 

User Request:  I want to find out if the patient with ID 10014729 underwent a resection surgery. If they did, I also want to obtain all their surgery times. If there is no record of a resection, then I don't need any further information.

Task Breakdown:
1. **Identify Resection Surgery Records:** Search the available data for any records indicating a resection surgery for patient ID 10014729.                                                                                             │
2. **Check for Resection Confirmation:** Determine if a resection surgery was actually performed based on the identified records.                                                                                                        │
3. **Extract Surgery Times (If Resection Confirmed):** If a resection surgery was performed, extract all recorded surgery times associated with that patient and the resection procedure.                                                │
4. **Report Findings:**  Present the findings. If a resection was performed, provide the surgery times. If no resection was found, state that no resection surgery was recorded for the patient.  

"""


class DecomposeTool:
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
    ):
        self.model = model
        self.temperature = temperature
        self.instruction = DECOMPOSITION_INSTRUCTION

    def invoke(self, user_request: str):
        agent_cost = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": user_request},
        ]

        while True:
            try:
                res = completion(
                    messages=messages,
                    model=self.model,
                    temperature=self.temperature,
                )
                agent_cost += res._hidden_params["response_cost"]
                break
            except Exception as e:
                time.sleep(3)
                print(e, end="\r")
        next_message = res.choices[0].message.model_dump()
        next_message["role"] = "user"
        logger.log_chat(next_message["content"], "Decompose Tool Response")
        return next_message


class PlanningAgent(Agent):
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
        self.instruction = TOOL_CALLING_INSTRUCTION + "\nRules:\n" + self.rule
        self.decompose_tool = DecomposeTool(
            model=model,
            temperature=temperature,
        )
        self.is_decompose = False

    def run(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> AgentRunResult:
        agent_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs_user = env_reset_res.observation
        env_info = env_reset_res.info.model_dump()
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": obs_user},
        ]

        logger.log_chat(obs_user, "User request")
        for _ in range(max_num_steps):
            while True:
                try:
                    res = completion(
                        messages=messages,
                        model=self.model,
                        tools=self.tools_info,
                        temperature=self.temperature,
                    )
                    agent_cost += res._hidden_params["response_cost"]
                    break
                except Exception as e:
                    time.sleep(3)
                    print(e, end="\r")
            next_message = res.choices[0].message.model_dump()
            action = convert_message_to_action(next_message)
            if action.name == "respond":
                logger.log_chat(next_message["content"], "Agent Response")
            else:
                logger.log_chat(
                    next_message["tool_calls"][0]["function"]["arguments"],
                    f"Tool Call : {action.name}",
                )

            env_response = env.step(action)
            reward = env_response.reward
            env_info = {**env_info, **env_response.info.model_dump()}
            if action.name != "respond":
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                )
                logger.log_chat(env_response.observation, "Tool Response")
            else:
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
                logger.log_chat(env_response.observation, "User Response")

                if not self.is_decompose:
                    next_message = self.decompose_tool.invoke(
                        user_request=env_response.observation
                    )
                    messages.pop()
                    messages.append(next_message)
                    self.is_decompose = True
                    continue

            if env_response.done:
                break

        return AgentRunResult(
            reward=reward,
            messages=messages,
            agent_cost=round(agent_cost, 8),
            info=env_info,
        )
