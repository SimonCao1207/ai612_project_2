import abc
from typing import Optional

from src.envs.base import Env
from src.types_utils import AgentRunResult


class Agent(abc.ABC):
    @abc.abstractmethod
    def run(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> AgentRunResult:
        raise NotImplementedError
