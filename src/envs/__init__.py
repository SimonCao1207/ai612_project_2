from typing import Optional

from src.envs.base import Env
from src.retrieve import VectorDB


def get_env(
    env_name: str,
    eval_mode: str,
    user_strategy: str,
    vector_db: VectorDB,
    user_model: Optional[str] = None,
    task_index: Optional[int] = None,
) -> Env:
    if env_name == "mimic_iv":
        from src.envs.mimic_iv import MimicIVEnv

        return MimicIVEnv(
            eval_mode=eval_mode,
            user_strategy=user_strategy,
            user_model=user_model,
            task_index=task_index,
            vector_db=vector_db,
        )
    else:
        raise ValueError(f"Unknown environment: {env_name}")
