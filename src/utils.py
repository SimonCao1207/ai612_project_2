import json
import re
from ast import literal_eval
from typing import Any, Dict

from src.types_utils import Action


def parse_sql(response: str) -> str:
    pattern = r"```sql([\s\S]*?)```"
    matches = re.findall(pattern, response)
    if matches:
        return matches[-1].strip()
    stripped = response.strip()
    if stripped.lower().startswith("select") or stripped.lower().startswith("with"):
        return stripped
    raise ValueError("No SQL found in the response")


def process_item(item):
    try:
        item = round(float(item), 3)
    except:
        pass
    return str(item)


def process_result(result):
    try:
        result = literal_eval(result)
    except:
        pass
    if type(result) == str:
        return result
    else:
        return sorted([[process_item(c) for c in row] for row in result])[
            :100
        ]  # only compare first 100 results


def convert_message_to_action(
    message: Dict[str, Any],
) -> Action:
    if (
        "tool_calls" in message
        and message["tool_calls"] is not None
        and len(message["tool_calls"]) > 0
        and message["tool_calls"][0]["function"] is not None
    ):
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(name="respond", kwargs={"content": message["content"]})
