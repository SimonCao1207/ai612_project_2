import abc
from litellm import completion
from typing import Optional, List, Dict, Any

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

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        res = completion(
            model=self.model, messages=messages, temperature=1.0
        )
        message = res.choices[0].message
        self.messages.append(message.model_dump())
        self.total_cost += res._hidden_params["response_cost"]
        return message.content

    def build_system_prompt(self, instruction: Optional[str]) -> str:
        instruction_display = (
            ("\n\nInstruction: " + instruction + "\n")
            if instruction is not None
            else ""
        )
        return f"""You are a human user who wants to retrieve data from an EHR database by interacting with an DB agent.
User instruction: {instruction_display}
Rules:
- Current time is 2100-12-31 23:59:00.
- You don't know SQL at all and only have a rough idea of what information the database contains.
- Explain your intent in plain language so that the DB agent understands exactly what you want.
- When making your initial request to the DB agent, do not reveal everything you want all at once.
- Do not be too wordy in your messages. Be concise and to the point.
- Only provide the information necessary to ask or respond to the DB agent.
- Do not hallucinate information that is not provided in the instructions.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Even if the DB agent transfers the task to you, you must not complete it yourself. You are reactive to the DB agent and only respond to its clarifying questions.
- If your goal is satisfied, generate '###END###' to end the conversation.
- Try to make the conversation as natural as possible, and stick to the user instruction.
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

def load_user(
    user_strategy: str,
    model: str
) -> BaseUser:
    if user_strategy == "llm":
        return LLMUser(model=model)
    raise ValueError(f"Unknown user strategy {user_strategy}")