import json
import re
import time
from typing import Any, Dict, List, Optional

from litellm import completion

from src.agents.base import Agent
from src.envs.base import Env
from src.log import BLUE_HEX, YELLOW_HEX, Logger
from src.types_utils import AgentRunResult, EnvInfo, RewardInfo
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
        style: str = YELLOW_HEX,
    ) -> tuple[str, float]:
        messages = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": initial_message},
        ]
        logger.log_chat(
            initial_message, f"User Initial Request (Task {task_index})", style=style
        )
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
                logger.log_rule(
                    f"Retrying InterrogatorAgent: cleared message cache (retry {retries})",
                    style=style,
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
                assistant_reply["content"],
                f"Interrogator Agent (Task {task_index})",
                style=style,
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

            logger.log_chat(user_reply, f"User Response Task {task_index}", style=style)
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
        actions = []

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
            actions.append(action)
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

        if reward == 0.0:
            final_instruction, _ = interrogator.run(
                env, obs_user, task_index=task_index, style=BLUE_HEX
            )
            refined_sql_agent = RefinedSQLAgent(
                goal=final_instruction,
                model=self.model,
                env=env,
                actions=actions,
                messages=messages,
                temperature=0.5,
            )
            final_messages, new_reward, reward_info = refined_sql_agent.run(
                task_index=task_index
            )
            reward = new_reward
            if reward_info.reward is not None:
                new_env_info = EnvInfo(task=env.task, reward_info=reward_info)
                env_info = {**env_info, **new_env_info.model_dump()}
        else:
            final_messages = messages
        return AgentRunResult(
            reward=reward,
            messages=final_messages,
            agent_cost=round(agent_cost, 8),
            info=env_info,
        )


def check_sql_query(
    query: str,
    response: str,
    goal: str,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.0,
) -> bool:
    """
    Use an LLM to check if the SQL query is relevant to the goal.
    Returns True if the LLM says the query matches the goal, else False.
    """

    system_prompt = (
        "You are an expert SQL assistant. Your job is to determine if a given SQL query and its response are directly relevant to a user's stated goal. "
        "Carefully review the user's goal, the SQL query, and the response. "
        "Reply only with 'yes' if the SQL query is clearly and directly attempting to fulfill the user's goal; otherwise, reply 'no'."
    )
    user_prompt = f"User goal: {goal}\n\nSQL query:\n{query}\n\nResponse: {response}\n\nDoes the SQL query match the goal? (yes/no):"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    while True:
        try:
            res = completion(
                messages=messages,
                model=model,
                temperature=temperature,
            )
            break
        except Exception as e:
            time.sleep(2)
            print(e, end="\r")
    answer = res.choices[0].message.content.strip().lower()
    return answer.startswith("yes")


class RefinedSQLAgent(Agent):
    def __init__(
        self,
        goal: str,
        model: str,
        env: Env,
        actions: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
    ):
        """
        This agent is trying to analyze the actions history and figure out which SQL query solve the user's request.
        Then it try to reason about the SQL and refines it till it is correct. Then it returns the new final messages with substituted SQL query.
        """
        self.model = model
        self.temperature = temperature
        self.goal = goal
        self.env = env
        self.messages = messages
        self.actions = actions

    def run(
        self,
        task_index: Optional[int] = None,
    ) -> tuple[List[Dict[str, Any]], Optional[float], RewardInfo]:
        sql_queries = []
        i = 0
        while i < len(self.messages):
            message = self.messages[i]
            action = convert_message_to_action(message)
            if action.name == "sql_db_query":
                if (
                    i + 1 < len(self.messages)
                    and self.messages[i + 1]["role"] == "tool"
                ):
                    response = self.messages[i + 1]["content"]
                    last_sql = {
                        "query": action.kwargs["query"],
                        "response": response,
                    }
                    logger.log_chat(
                        f"Checking SQL query: {action.kwargs['query']}\nResponse: {response}",
                        f"RefinedSQLAgent SQL Check (Task {task_index})",
                        style=BLUE_HEX,
                    )
                    # Filter out error responses
                    if isinstance(response, str) and (
                        "error" in response.lower() or "traceback" in response.lower()
                    ):
                        i += 2
                        continue
                    if check_sql_query(
                        action.kwargs["query"],
                        response,
                        self.goal,
                        model=self.model,
                        temperature=self.temperature,
                    ):
                        logger.log_code(
                            f"Relevant SQL (Task {task_index})",
                            action.kwargs["query"],
                        )
                        sql_queries.append(
                            {
                                "query": action.kwargs["query"],
                                "response": response,
                            }
                        )
                        i += 2
                        continue
            i += 1

        if len(sql_queries) == 0 and last_sql is not None:
            sql_queries = [last_sql]
        else:
            return self.messages, 0.0, RewardInfo(reward=None)

        # Build a concise, clear prompt for the LLM to improve the SQL query
        system_prompt = (
            "You are an expert SQL assistant. Your task is to help improve SQL queries for a given user goal. "
            "You will be provided with the user's goal and a list of previous SQL queries (with their responses) that failed to fully satisfy the goal. "
            "Carefully analyze the mistakes or missing elements in the previous queries and responses. "
            "Then, generate a single, corrected SQL query that best fulfills the user's goal, using all available information. "
            "Do NOT include any explanations, comments, or formatting—just output the improved SQL query itself."
        )
        user_prompt = f"User goal: {self.goal}\n\n"
        user_prompt += "Previous SQL queries and their responses:\n"
        for idx, sql in enumerate(sql_queries, 1):
            user_prompt += (
                f"{idx}. SQL query:\n{sql['query']}\nResponse:\n{sql['response']}\n\n"
            )
        user_prompt += "Please provide only the improved SQL query (do not include explanations or comments):\nSQL query:"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        logger.log_chat(
            user_prompt, f"Refined SQL Prompt (Task {task_index})", style=BLUE_HEX
        )

        while True:
            try:
                res = completion(
                    messages=messages,
                    model=self.model,
                    temperature=self.temperature,
                    reasoning_effort="high",
                )
                break
            except Exception as e:
                logger.log_chat(
                    f"LLM completion error: {e}",
                    f"RefinedSQLAgent LLM Error (Task {task_index})",
                    style=BLUE_HEX,
                )
                time.sleep(2)
                print(e, end="\r")
        improved_sql = res.choices[0].message.content.strip()

        def post_process_sql(sql: str) -> str:
            sql = sql.strip()
            if sql.startswith("```"):
                lines = sql.splitlines()
                if lines and lines[0].strip().startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                sql = "\n".join(lines).strip()
            return sql

        improved_sql = post_process_sql(improved_sql)

        logger.log_chat(
            improved_sql, f"Refined SQL Query (Task {task_index})", style=BLUE_HEX
        )

        improved_message = {
            "role": "assistant",
            "content": improved_sql,
            "tool_calls": [
                {
                    "id": "refined_sql",
                    "function": {
                        "name": "sql_db_query",
                        "arguments": json.dumps({"query": improved_sql}),
                    },
                }
            ],
        }
        env_response = self.env.step(convert_message_to_action(improved_message))
        reward_res = self.env.calculate_reward_sql()
        reward = reward_res.reward
        reward_info = reward_res
        tool_response_message = {
            "role": "tool",
            "tool_call_id": "refined_sql",
            "name": "sql_db_query",
            "content": env_response.observation,
        }

        logger.log_chat(
            env_response.observation,
            f"Tool Response for Refined SQL (Task {task_index})",
            style=BLUE_HEX,
        )

        if self.messages and self.messages[-1].get("role") == "user":
            return (
                (
                    self.messages[:-1]
                    + [improved_message, tool_response_message]
                    + [self.messages[-1]]
                ),
                reward,
                reward_info,
            )
        else:
            self.messages.extend(
                [
                    improved_message,
                    tool_response_message,
                ]
            )
            return (
                self.messages + [improved_message, tool_response_message],
                reward,
                reward_info,
            )
