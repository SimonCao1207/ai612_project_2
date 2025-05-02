import json
import os
import threading
import traceback
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from math import comb
from typing import List

file_lock = threading.Lock()

from dotenv import load_dotenv

from automatic_evaluation import role_fault_classification
from src.agent_factory import get_agent
from src.envs import get_env
from src.types import CostInfo, EnvRunResult

load_dotenv()


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=["mimic_iv"],
        help="Environment name for fetching user instructions",
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        required=True,
        choices=["valid", "test"],
        help="Task set to use",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="The agent model to use"
    )
    parser.add_argument(
        "--agent_strategy",
        type=str,
        required=True,
        choices=["tool-calling", "custom"],
        help="The agent strategy to use",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=True,
        help="Sampling temperature for the action model",
    )
    parser.add_argument(
        "--user_model", type=str, default="gpt-4o-mini", help="The user model to use"
    )
    parser.add_argument(
        "--user_strategy", type=str, default="llm", help="The user strategy to use"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results",
        help="Directory to save the results",
    )
    parser.add_argument(
        "--seed", type=int, required=False, default=42, help="Seed for reproducibility"
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        required=False,
        default=1,
        help="Number of trials to run",
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        required=False,
        default=1,
        help="Maximum concurrency level",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        required=False,
        default=0,
        help="Start index for tasks",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        required=False,
        default=-1,
        help="End index for tasks (-1 for all)",
    )
    parser.add_argument(
        "--task_ids",
        nargs="+",
        type=int,
        required=False,
        default=None,
        help="Specific task ids to run",
    )
    parser.add_argument(
        "--simulation_retry",
        type=int,
        required=False,
        default=3,
        help="Number of simulation retries",
    )
    return parser.parse_args()


def display_metrics(results: List[EnvRunResult]) -> None:
    """Compute and display average reward and pass@k/pass^k metrics."""

    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    unique_trials = len(set(r.trial for r in results))
    rewards = [r.reward for r in results]
    avg_reward = round(sum(rewards) / len(rewards) * 100, 1)

    success_counts: dict[int, int] = {}
    for res in results:
        success_counts[res.task_idx] = success_counts.get(res.task_idx, 0) + (
            1 if is_successful(res.reward) else 0
        )

    pass_at_k = {}
    for k in range(1, unique_trials + 1):
        pass_k_total = sum(
            comb(unique_trials - success, k) / comb(unique_trials, k)
            for success in success_counts.values()
        )
        pass_at_k[k] = round((1 - (pass_k_total / len(success_counts))) * 100, 3)

    pass_hat_k = {}
    for k in range(1, unique_trials + 1):
        pass_k_total = sum(
            comb(success, k) / comb(unique_trials, k)
            for success in success_counts.values()
        )
        pass_hat_k[k] = round((pass_k_total / len(success_counts)) * 100, 3)

    print(f"📈 Pass@4: {pass_at_k[4]} (%)")
    print(f"📈 Pass^4: {pass_hat_k[4]} (%)")
    print(f"🏆 Final Score: {(pass_at_k[4] + pass_hat_k[4]) / 2} (%)")


def update_checkpoint(ckpt_path, result, lock):
    with lock:
        data = []
        if os.path.exists(ckpt_path):
            with open(ckpt_path, "r") as f:
                data = json.load(f)
        with open(ckpt_path, "w") as f:
            json.dump(data + [result.model_dump()], f, indent=2)


def run(config: Namespace) -> List[EnvRunResult]:
    timestamp = datetime.now().strftime("%m%d%H%M%S")
    checkpoint_filename = (
        f"{config.env}-{config.agent_strategy}-{os.path.basename(config.model)}-{config.temperature}_"
        f"range_{config.start_index}-{config.end_index}_user-{config.user_model}-{config.user_strategy}_{timestamp}_{config.eval_mode}.json"
    )
    ckpt_path = os.path.join(config.result_dir, checkpoint_filename)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    print(f"Loading user with strategy: {config.user_strategy}")

    env = get_env(
        env_name=config.env,
        eval_mode=config.eval_mode,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
    )
    agent = get_agent(
        tools_info=env.tools_info,
        model=config.model,
        temperature=config.temperature,
        agent_strategy=config.agent_strategy,
        rule=env.rule,
    )

    total_tasks = len(env.tasks)
    end_index = (
        total_tasks if config.end_index == -1 else min(config.end_index, total_tasks)
    )
    results: List[EnvRunResult] = []
    lock = threading.Lock()

    if config.task_ids:
        print(f"Running tasks: {config.task_ids} (checkpoint path: {ckpt_path})")
    else:
        print(
            f"Running tasks: {config.start_index} to {end_index} (checkpoint path: {ckpt_path})"
        )

    if config.task_ids:
        idx = config.task_ids
    else:
        idx = list(range(config.start_index, end_index))
    idx_to_run = idx * config.num_trials
    trials = [i for i in range(1, config.num_trials + 1) for _ in idx]

    def _run(idx: int, trial: int) -> EnvRunResult:
        simulation_retry = 0
        isolated_env = get_env(
            env_name=config.env,
            eval_mode=config.eval_mode,
            user_strategy=config.user_strategy,
            user_model=config.user_model,
            task_index=idx,
        )
        exit_flag = False
        while True:
            try:
                response = agent.run(env=isolated_env, task_index=idx)
                fault_result = role_fault_classification(
                    {
                        "messages": response.messages,
                        "instruction": isolated_env.task.instruction,
                        "gold_sql": isolated_env.task.gold_sql,
                        "gold_answer": isolated_env.task.gold_answer,
                    }
                )
                if config.eval_mode == "valid" and response.reward is None:
                    response.reward = 0.0
                result = EnvRunResult(
                    task_idx=idx,
                    trial=trial,
                    reward=response.reward,
                    info=response.info,
                    messages=response.messages,
                    cost=CostInfo(
                        agent_cost=response.agent_cost,
                        user_cost=isolated_env.user.get_total_cost(),
                        eval_cost=fault_result["eval_cost"],
                        total_cost=round(
                            response.agent_cost
                            + isolated_env.user.get_total_cost()
                            + fault_result["eval_cost"],
                            8,
                        ),
                    ),
                )
                simulation_retry += 1
                if (
                    fault_result["role"] == "agent"
                    or simulation_retry == config.simulation_retry
                ):
                    update_checkpoint(ckpt_path, result, lock)
                    exit_flag = True
            except Exception as e:
                result = EnvRunResult(
                    task_idx=idx,
                    trial=trial,
                    reward=0.0,
                    info={"error": str(e), "traceback": traceback.format_exc()},
                    messages=[],
                    cost=CostInfo(),
                )
            if exit_flag:
                break
            print(
                f"Retrying... {simulation_retry}/{config.simulation_retry}",
                f"task_id={idx}",
                result.info,
            )
        if config.eval_mode == "valid":
            print("✅" if result.reward == 1 else "❌", f"task_id={idx}", result.info)
            print("-----")
        return result

    with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
        res = list(executor.map(_run, idx_to_run, trials))
        results.extend(res)

    if config.eval_mode == "valid":
        display_metrics(results)


if __name__ == "__main__":
    config = parse_arguments()
    run(config)
