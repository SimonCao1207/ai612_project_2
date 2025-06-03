from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Task(BaseModel):
    task_id: str
    instruction: str
    gold_sql: Optional[str] = None
    gold_answer: Optional[List[Any]] = None


class Action(BaseModel):
    name: str
    kwargs: Dict[str, Any]


class Tool(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str]


class RewardInfo(BaseModel):
    reward: Optional[float] = None
    info: Optional[Dict[str, Any]] = None


class EnvInfo(BaseModel):
    task: Task
    reward_info: RewardInfo


class EnvResponse(BaseModel):
    observation: str
    reward: Optional[float] = None
    done: bool
    info: EnvInfo


class AgentRunResult(BaseModel):
    reward: Optional[float] = None
    messages: List[Dict[str, Any]]
    agent_cost: Optional[float] = None
    info: Dict[str, Any]


class CostInfo(BaseModel):
    agent_cost: Optional[float] = None
    user_cost: Optional[float] = None
    eval_cost: Optional[float] = None
    total_cost: Optional[float] = None


class EnvRunResult(BaseModel):
    task_idx: int
    trial: int
    reward: Optional[float] = None
    info: Dict[str, Any]
    messages: List[Dict[str, Any]]
    cost: CostInfo
