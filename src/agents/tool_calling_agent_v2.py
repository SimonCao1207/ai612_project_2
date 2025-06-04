import re
import time
from typing import Any, Dict, List, Optional

from litellm import completion

from src.agents.base import Agent
from src.envs.base import Env
from src.log import Logger
from src.types_utils import AgentRunResult
from src.utils import convert_message_to_action

logger = Logger()

TOOL_CALLING_INSTRUCTION = """- You are a SQL agent that translates natural language questions into precise SQL queries for electronic health records (EHR).
- You are currently engaged in a conversation with a user who wants to retrieve data from an EHR database.
- You can interact with the database to learn more about its schema or the values stored in it by using the tools provided.
- Do not invent or fabricate any information not provided by the user or the tools.
- You should make at most one tool call at a time.
- If you do call a tool, do not respond to the user in that same turn.
- Before generating any SQL query, you must use the tool `instruction_sql_search` to extract samples similar to the user request for reference. Carefully review the retrieved samples and use them to guide your SQL generation.
- Do not generate SQL queries directly without knowing the database schema and values intended to be used in the SQL query by calling substring_search_tool.
- When the user asks for specific diagnoses, procedures, medications, or lab tests, try your best to use the tool to search for relevant information in the database and determine if it relates to the user's request.
- Only when you have gathered all necessary information from the user or the database, produce a single, valid SQL query that fully reflects the user's request.
- Avoid partial or speculative queries that could cause confusion or yield inaccurate results.
"""

INTERROGATOR_INSTRUCTION = """
You are a helpful assistant that helps users clarify their requests to retrieve information from an electronic health record (EHR) database.

The user is a human with no technical background or knowledge of SQL or the database schema.

Your role is to guide the user to express their intent clearly and completely in natural language.

## Conversation Strategy

1. When a user gives an ambiguous or partial request, ask **simple, goal-related follow-up questions** to help them clarify what they want.
    - Do not ask about SQL, database schema, or structure.
    - Do not use technical terms.
    - Keep questions open-ended, like:
        - “Is there anything else I should know?”
        - “Do you have a specific patient in mind?”
        - “What exactly do you want to find out?”

2. Repeat step 1 until the user indicates they have provided all the information.

3. When the user indicates they are finished (e.g., says “no, that's all,” “goodbye,” or anything similar to ending the conversation), respond with:

---
Please confirm your final request by copying and pasting it inside the following tags, **and nothing else**:

Example:
<final_instruction>Find all medications prescribed for patients with high blood pressure.</final_instruction>

✅ Please reply with only your instruction wrapped inside <final_instruction> tags, like in the example above.
❌ Do not add explanations, greetings, or any other text outside the tags.
---

4. Wait for the user to respond with the final instruction in that format. Do not produce the instruction yourself.

## Important Notes

- Do NOT use SQL terminology or generate SQL queries.
- Do NOT try to guess the schema or data.
- Only focus on understanding the user's intent in plain language.
- Be friendly, helpful, and non-technical.
"""


class InterrogatorAgent(Agent):
    def __init__(self, model: str, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.instruction = INTERROGATOR_INSTRUCTION

    def run(
        self,
        env: Env,
        initial_message: str,
        max_num_steps: int = 30,
        task_index: Optional[int] = None,
    ) -> tuple[str, float]:
        messages = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": initial_message},
        ]
        logger.log_chat(initial_message, f"User Initial Request (Task {task_index})")
        confirmed = False
        agent_cost = 0.0
        trial = 0
        max_retries = 3
        retries = 0

        def extract_final_instruction(messages) -> Optional[str]:
            for message in messages:
                content = message["content"]
                # Normalize invisible characters
                content = content.replace("\r", "").replace("\xa0", " ").strip()

                match = re.search(
                    r"<final_instruction>(.*?)</final_instruction>",
                    content,
                    re.DOTALL | re.IGNORECASE,
                )
                if match:
                    return match.group(1).strip()
            return None

        while not confirmed:
            if trial >= max_num_steps:
                if retries >= max_retries:
                    raise RuntimeError(
                        "Maximum number of steps reached and retry limit exceeded without confirmation."
                    )
                # Reset conversation state
                messages = [
                    {"role": "system", "content": self.instruction},
                    {"role": "user", "content": initial_message},
                ]
                trial = 0
                retries += 1
                logger.log_chat(
                    f"Retrying InterrogatorAgent: cleared message cache (retry {retries})",
                    f"Interrogator Agent (Task {task_index})",
                )
                continue

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
                    time.sleep(2)
                    print(e, end="\r")

            assistant_reply = res.choices[0].message.model_dump()
            logger.log_chat(
                assistant_reply["content"], f"Interrogator Agent (Task {task_index})"
            )
            action = convert_message_to_action(assistant_reply)
            env_response = env.step(action)

            user_reply = env_response.observation
            messages.extend(
                [
                    assistant_reply,
                    {"role": "user", "content": user_reply},
                ]
            )

            logger.log_chat(user_reply, f"User Response Task {task_index}")
            test_messages = [messages[-1], messages[-2]]

            extracted_instruction = extract_final_instruction(test_messages)

            # Check if the user provided a valid final instruction
            if extracted_instruction is not None:
                confirmed = True

            trial += 1

        final_instruction = extracted_instruction
        return final_instruction or messages[-1]["content"], agent_cost


class ToolCallingAgentV2(Agent):
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

    def run(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> AgentRunResult:
        agent_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs_user = env_reset_res.observation
        env_info = env_reset_res.info.model_dump()
        reward = 0.0

        interrogator = InterrogatorAgent(model=self.model, temperature=self.temperature)
        final_instruction, cost = interrogator.run(env, obs_user, task_index=task_index)
        agent_cost += cost

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": final_instruction},
        ]

        logger.log_chat(final_instruction, f"User Final Request (Task {task_index})")

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
                logger.log_chat(
                    next_message["content"],
                    f"Agent Response Task {task_index}",
                )
            else:
                logger.log_chat(
                    next_message["tool_calls"][0]["function"]["arguments"],
                    f"Tool Call : {action.name} (Task {task_index})",
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
                logger.log_chat(
                    env_response.observation, f"Tool Response Task {task_index}"
                )
            else:
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
                logger.log_chat(
                    env_response.observation, f"User Response Task {task_index}"
                )
            if env_response.done:
                break

        return AgentRunResult(
            reward=reward,
            messages=messages,
            agent_cost=round(agent_cost, 8),
            info=env_info,
        )
