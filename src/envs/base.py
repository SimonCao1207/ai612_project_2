import random
import sqlite3
from typing import Dict, List, Optional, Type

from src.envs.user import load_user
from src.types import (
    Action,
    EnvInfo,
    EnvResponse,
    RewardInfo,
    Task,
    Tool,
)
from src.utils import process_result


class Env(object):
    def __init__(
        self,
        tools: List[Type[Tool]],
        tasks: List[Task],
        user_strategy: str,
        user_model: str,
        db_path: str,
        task_index: Optional[int] = None,
        rule: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.tools_map: Dict[str, Type[Tool]] = {
            tool.get_info()["function"]["name"]: tool for tool in tools
        }
        self.tools_info = [tool.get_info() for tool in tools]
        self.tasks = tasks
        if task_index is not None:
            self.task_index = task_index
        else:
            self.task_index = random.randint(0, len(tasks) - 1)
        self.task = tasks[self.task_index]
        self.rule = rule
        self.user = load_user(user_strategy=user_strategy, model=user_model)
        self.actions: List[Action] = []
        self.db_path = db_path

    def reset(self, task_index: Optional[int] = None) -> EnvResponse:
        if task_index is None:
            task_index = random.randint(0, len(self.tasks) - 1)
        self.task_index = task_index
        self.task = self.tasks[task_index]
        self.actions = []
        initial_observation = self.user.reset(instruction=self.task.instruction)
        return EnvResponse(
            observation=initial_observation,
            reward=0.0,
            done=False,
            info=EnvInfo(task=self.task, reward_info=RewardInfo()),
        )

    def step(self, action: Action) -> EnvResponse:
        self.actions.append(action)

        info = EnvInfo(task=self.task, reward_info=RewardInfo())
        reward = 0.0
        done = False
        if action.name == "respond":
            observation = self.user.step(action.kwargs["content"])
            done = "###END###" in observation
        elif action.name in self.tools_map:
            try:
                observation = self.tools_map[action.name].invoke(**action.kwargs)
            except Exception as e:
                observation = f"Error: {e}"
        else:
            observation = f"Unknown action {action.name}"
        if done:
            reward_res = self.calculate_reward_sql()
            reward = reward_res.reward
            info.reward_info = reward_res
        return EnvResponse(observation=observation, reward=reward, done=done, info=info)

    def calculate_reward_sql(self) -> RewardInfo:
        if self.task.gold_sql is None:
            return RewardInfo(reward=None, info={"pred_sql": None, "pred_answer": None})

        reward = 0.0

        # Find the last SQL action
        last_sql = None
        for action in self.actions:
            if action.name == "sql_db_query":
                last_sql = action
        if last_sql:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            try:
                gold_answer = process_result(self.task.gold_answer)
                cursor.execute(last_sql.kwargs["query"])
                pred_sql_answer = cursor.fetchall()
                pred_sql_answer = process_result(pred_sql_answer)
                # returning a single column
                if pred_sql_answer and len(pred_sql_answer) > 0:
                    if len(pred_sql_answer[0]) == 1:
                        if pred_sql_answer == gold_answer:
                            reward = 1.0
                    # returning multiple columns
                    else:
                        converted_pred_sql_answer = list(zip(*pred_sql_answer))
                        for i in range(len(converted_pred_sql_answer)):
                            if sorted(set(converted_pred_sql_answer[i])) == sorted(
                                set([el[0] for el in gold_answer])
                            ):
                                reward = 1.0
                                break
            except sqlite3.Error:
                pred_sql_answer = []
            conn.close()
            reward_info = RewardInfo(
                reward=reward,
                info={
                    "pred_sql": last_sql.kwargs["query"],
                    "pred_answer": pred_sql_answer,
                },
            )
        else:
            reward_info = RewardInfo(
                reward=reward, info={"pred_sql": None, "pred_answer": None}
            )
        return reward_info
