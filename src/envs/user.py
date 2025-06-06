import abc
import time
from typing import Any, Dict, List, Optional

from litellm import completion
from litellm.exceptions import ContextWindowExceededError


class BaseUser(abc.ABC):
    @abc.abstractmethod
    def reset(self, instruction: Optional[str] = None) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, content: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_total_cost(self) -> float:
        raise NotImplementedError


class LLMUser(BaseUser):
    def __init__(self, model: str) -> None:
        super().__init__()
        self.messages: List[Dict[str, Any]] = []
        self.model = model
        self.total_cost = 0.0
        self.reset()

    # def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
    #    res = completion(
    #        model=self.model, messages=messages, temperature=0.5
    #    )
    #    message = res.choices[0].message
    #    self.messages.append(message.model_dump())
    #    self.total_cost += res._hidden_params["response_cost"]
    #    return message.content

    ####
    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        while True:
            try:
                res = completion(model=self.model, messages=messages, temperature=0.5)
                message = res.choices[0].message
                if (
                    len(self.messages) > 2
                    and message.content
                    and (
                        "SELECT" in message.content
                        or "default_api" in message.content
                        or "print(" in message.content
                    )
                ):
                    messages += [
                        {
                            "role": "user",
                            "content": "You must act like a user, not like a DB agent. Do not generate any SQL query or other DB agent artifacts like default_api or print() in your response, but follow the instruction and rules in the system prompt.",
                        }
                    ]
                    raise ValueError(
                        "The user acts like a DB agent: the user response includes SQL query or other DB agent artifacts like default_api or print()"
                    )
                self.messages.append(message.model_dump())
                self.total_cost += res._hidden_params["response_cost"]
                return message.content
            except ContextWindowExceededError:
                # print("⚠️ Context window exceeded:", e)
                return "###END###"
            except Exception:
                # print(f"[User] Error: {e}")
                time.sleep(3)

    ####

    def build_system_prompt(self, instruction: Optional[str]) -> str:
        instruction_display = (
            ("\n\nInstruction: " + instruction + "\n")
            if instruction is not None
            else ""
        )
        return f"""You are a human user who wants to retrieve data from an EHR database by interacting with an DB agent.
The important thing is that, you have no background knowledge about the Structured Query Language. You should not act like a person who knows SQL.
In other words, you are a person that asks a DB agent with a natural language, and you only understand the execution result of the SQL query the DB agent executes.
User instruction: {instruction_display}

[VERY IMPORTANT RULES]
1. The current time is 2100-12-31 23:59:00.
2. You don't know SQL and have only a general understanding of the database contents.
3. Explain your question in plain language so that the DB agent understands what you want.
4. Keep your messages short and to the point, avoiding words like "please" or "thank you."
5. Start with a short, vague question to convey your goal in the first turn.
6. Don't reveal all your needs at once. Share one or two sentences per turn.
7. Don't repeat the user instruction. Use your own words to describe what you need.
8. You don't know the form in which the requested data is stored. If the DB agent cannot retrieve the data you want, ask it to explore the database for their correct format.
9. Only provide information relevant to asking or replying to the DB agent.
10. Don't make up or assume extra information or assumptions not given in the user instruction.
11. Even if the DB agent hands the task back to you (e.g., retrieving the database schema, generating or executing SQL queries, reviewing SQL queries, or using API calls), do not attempt to complete the task yourself. You lack access to external resources, so instruct the DB agent to handle the entire task. Respond only to the agent's clarifications based on the user instruction.
12. You are the user adhering to the user instructions, seeking help from the database agent because you don't have the expertise in SQL or database. You are not assisting the agent, but the agent is assisting you to complete the task.
13. If the DB agent gives a partial answer, don't complete it yourself. Ask the agent to complete it (e.g., phrasing answers aligned with the user instruction or simple calculation, etc).
14. At the end, ask the DB agent to check if your question is fully satisfied as every detail conditions stated in the user instruction. If not, ask for the remaining parts.
15. If the answer satisfies your goal completely, generate `###END###` to end the conversation.
16. Do not ask questions beyond the user instructions. Even if the DB agent says feel free to ask, stick to the instructions provided (e.g., do not request additional timestamps or any other information not specified).
17. End the conversation immediately after the user instruction is fully satisfied.
18. Keep the conversation natural, not robotic.
19. If not specified in the user instruction, you should consider all the values the DB agent retrieves.
20. Do not respond with a fake execution result of the SQL query, or made up information about the database schema.
21. NEVER provide information you don't have access to e.g., information related to the database schema, database contents, execution result of the SQL query, etc.
22. If the DB agent replies only with the SQL query, you should say "I have no knowledge about the Structured Query Language. Give me the explanation or the result of the SQL query."
23. Whenever asked about the information about the database or SQL, you should say "You should find that out for me."
"""

    def reset(self, instruction: Optional[str] = None) -> str:
        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(instruction=instruction),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        new_message = self.generate_next_message(self.messages)
        return new_message

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        new_message = self.generate_next_message(self.messages)
        return new_message

    def get_total_cost(self) -> float:
        return round(self.total_cost, 8)


def load_user(user_strategy: str, model: str) -> BaseUser:
    if user_strategy == "llm":
        return LLMUser(model=model)
    raise ValueError(f"Unknown user strategy {user_strategy}")
