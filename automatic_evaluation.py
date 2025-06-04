import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import pandas as pd
from dotenv import load_dotenv
from litellm import completion

from src.types_utils import Task

load_dotenv()
MODEL = "gemini/gemini-2.0-flash"

role_prompt = """Your task is to identify which entity was responsible for the initial mistake that led to the failure of EHR database question answering.
- The conversation below is between a user and a database agent.
- The role of the simulated user is to follow the user instruction to ask questions about the data stored in the EHR database.
- The role of the agent is to help users to retrieve and answer questions that the user requests.
- The role of the environment is to provide the precise user instruction and evaluation (checking only the agent utterance as answers).
- The conversation has been flagged as failed because one of the entities first made a mistake that caused the task failure.

The entity responsible for the task failure is one of the following:
- User errors:
    - The user does not follow the user instruction, generating requests that are unrelated or fabricate information beyond it.
    - The user ends the conversation prematurely, preventing the agent from fulfilling the request.
    - The user continues the conversation beyond the user instruction so that answers beyond the user instruction are evaluated.
    - The user makes mistakes that are not mentioned above.
- Agent errors:
    - The agent generates SQL with incorrect logic or structure, failing to reflect the user's request (e.g., wrong joins, incorrect clauses).
    - The agent generates SQL with incorrect data values stored in the EHR (e.g., wrong diagnosis names, drug names compared to values in gold SQL, etc).
    - The agent does not generate SQL or at all during the conversation (not using sql_db_query not even once during the conversation).
    - The agent makes mistakes that are not mentioned above.
- Environment errors:
    - All other errors not mentioned above.

First read the user instruction below and carefully read through the conversation from the beginning to identify which entity cause the initial mistake that led to the task failure.

----- start user instruction -----
{instruction}
----- end user instruction -----

----- start gold SQL -----
{gold_sql}
----- end gold SQL -----

----- start gold answer -----
{gold_answer}
----- end gold answer -----

----- start conversation -----
{conversation}
----- end conversation -----

Response format:
{{
    "chain_of_thought": "step-by-step reasoning for which entity is responsible for the initial fault that led to the task failure",
    "role": "entity initially responsible for the task failure (user, agent, or environment)"
}}
"""


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=["mimic_iv", "eicu"],
        help="Environment name for fetching user instructions",
    )
    parser.add_argument(
        "--results_path", type=str, required=True, help="Path to the results file"
    )
    parser.add_argument(
        "--max_concurrency", type=int, default=1, help="Maximum concurrency level"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to the output file"
    )
    parser.add_argument(
        "--max_num_failed_results",
        "-n",
        type=int,
        default=None,
        help="Maximum number of failed results to analyze",
    )
    return parser.parse_args()


def display_conversation(messages):
    if len(messages) == 0:
        raise ValueError("Trajectory is empty")
    log = []
    for item in messages:
        if item["role"] == "system":
            log.append(f"[{item['role'].capitalize()}]: {item['content']}")
        elif item["role"] == "user":
            log.append(f"[{item['role'].capitalize()}]: {item['content']}")
        elif item["role"] == "assistant":
            if item["content"]:
                log.append(f"[{item['role'].capitalize()}]: {item['content']}")
            else:
                for tool_call in item["tool_calls"]:
                    log.append(
                        f"[{item['role'].capitalize()}]: {tool_call['function']['name']}({tool_call['function']['arguments']})"
                    )
        elif item["role"] == "tool":
            log.append(f"[{item['role'].capitalize()}]: {item['content']}")
        else:
            raise ValueError(f"Unknown role: {item['role']}")
    return "\n".join(log)


def role_fault_classification(result: Dict[str, Any]) -> Dict[str, Any]:
    conversation_str = display_conversation(result["messages"])
    formatted_prompt = role_prompt.format(
        instruction=result["instruction"],
        gold_sql=result["gold_sql"],
        gold_answer=result["gold_answer"],
        conversation=conversation_str,
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": formatted_prompt},
    ]
    response = completion(messages=messages, model=MODEL, temperature=0.0)
    res = json.loads(response.choices[0].message.content)
    res["eval_cost"] = response._hidden_params["response_cost"]
    if "task_id" in result:
        res["task_id"] = result["task_id"]
    if "trial" in result:
        res["trial"] = result["trial"]
    return res


def main() -> None:
    args = get_args()
    with open(args.results_path, "r") as f:
        loaded_results = json.load(f)
    print(f"Loaded {len(loaded_results)} results")
    env = args.env
    with open(f"src/envs/{env}/{args.eval_mode}_data.json", "r") as f:
        tasks = [Task(**kwargs) for kwargs in json.load(f)]
    TASK_SET = {t.task_id: t for t in tasks}
    failed_results = [r for r in loaded_results if r["reward"] == 0.0]
    print(f"Found {len(failed_results)} failed dialog messages")
    if (
        pd.notnull(args.max_num_failed_results)
        and len(failed_results) > args.max_num_failed_results
    ):
        print(f"Limiting to {args.max_num_failed_results} failed dialog messages")
        failed_results = failed_results[: args.max_num_failed_results]

    results = []
    for result in failed_results:
        task_id = result["task_id"]
        trial = result["trial"]
        task = TASK_SET[task_id]
        instruction = task.instruction
        messages = result["messages"]
        gold_sql = task.gold_sql
        gold_answer = task.gold_answer
        results.append(
            {
                "task_id": task_id,
                "trial": trial,
                "instruction": instruction,
                "gold_sql": gold_sql,
                "gold_answer": gold_answer,
                "messages": messages,
            }
        )

    print(
        f"Valid samples: {sum(1 for r in results if r['messages'] is not None)} / Invalid samples: {sum(1 for r in results if r['messages'] is None)}"
    )

    with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        role_results = list(executor.map(role_fault_classification, results))

    total = len(role_results)
    user_count = sum(1 for r in role_results if r["role"].lower() == "user")
    agent_count = sum(1 for r in role_results if r["role"].lower() == "agent")
    env_count = sum(1 for r in role_results if r["role"].lower() == "environment")

    print(f"Reviewed {total} messages:\n")
    print("fault distribution:")
    print(f"  - User: {user_count} ({round(user_count / total * 100, 2)}%)")
    print(f"  - Agent: {agent_count} ({round(agent_count / total * 100, 2)}%)")
    print(f"  - Env (others): {env_count} ({round(env_count / total * 100, 2)}%)\n")

    # # Save the analyses to the output file.
    output_data = {}
    for r in role_results:
        output_data[f"{r['task_id']}-{r['trial']}"] = r

    os.makedirs("analysis", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Saved results to {args.output_path}")


if __name__ == "__main__":
    main()
