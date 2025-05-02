# streamlit run visualizer.py --server.port 8505

import json
import os
import re

import sqlparse
import streamlit as st

DATA_DIR = "results"


# @st.cache_data
def load_json_files(directory):
    json_files = []
    for f in os.listdir(directory):
        if f.endswith(".json"):
            # Extract the date from filename (last part after underscore)
            date_match = re.search(r"_(\d{10})_(valid|test)\.json$", f)
            if date_match:
                date_str = date_match.group(1)
                json_files.append((f, date_str))
            else:
                # Fallback to creation time if pattern not found
                json_files.append((f, os.path.getctime(os.path.join(directory, f))))

    # Sort by the extracted date string (or creation time if date not found)
    json_files.sort(key=lambda x: x[1])
    return [f[0] for f in json_files]


# @st.cache_data
def load_json(filepath):
    """Load and return JSON content from the given file."""
    with open(filepath, "r") as f:
        return json.load(f)


st.title("User-Assistant Interaction Logs Viewer")

st.sidebar.header("JSON Files")
files = load_json_files(DATA_DIR)

if not files:
    st.sidebar.write("No JSON files found in the data folder.")
else:
    selected_file = st.sidebar.selectbox("Choose a file", files)
    file_path = os.path.join(DATA_DIR, selected_file)

    st.markdown(
        f'<h3 style="font-size:20px;">Viewing: {selected_file}</h3>',
        unsafe_allow_html=True,
    )
    data = load_json(file_path)

    if isinstance(data, list):
        task_indices = sorted(set([element["task_idx"] for element in data]))
        # precompute rewards
        rewards = {}
        for element in data:
            task_idx = element["task_idx"]
            if task_idx not in rewards:
                rewards[task_idx] = []
            rewards[task_idx].append(element["reward"])

        # check if rewards are all None / rewards[idx] is a list of None
        test_mode = all(
            [all([r is None for r in rewards[idx]]) for idx in task_indices]
        )
        if not test_mode:
            rewards = {
                task_idx: [r if r is not None else 0 for r in rewards[task_idx]]
                for task_idx in task_indices
            }
            avg_rewards = {
                task_idx: round(
                    sum([r for r in rewards[task_idx]])
                    / len([r for r in rewards[task_idx]])
                    * 100,
                    1,
                )
                for task_idx in task_indices
            }
            # st.markdown(f'<h3 style="font-size:20px;">{len([1 for task_idx in task_indices if pd.notnull(avg_rewards[task_idx]) and avg_rewards[task_idx] > 0])} samples are at least correct once ({len(task_indices)} in total)</h3>', unsafe_allow_html=True)
            task_indices_with_rewards = [
                f"{task_idx} | Avg: {avg_rewards[task_idx]}% [{int(sum([r for r in rewards[task_idx]]))}/{len([r for r in rewards[task_idx]])}]"
                for task_idx in task_indices
            ]
            selected_task_idx = st.sidebar.selectbox(
                "Choose a Task ID", task_indices_with_rewards
            )
            selected_task_idx = int(selected_task_idx.split(" ")[0])
        else:
            selected_task_idx = st.sidebar.selectbox("Choose a Task ID", task_indices)
        matching_elements = [
            element for element in data if element.get("task_idx") == selected_task_idx
        ]

        if matching_elements:
            if not test_mode:
                average_reward = sum([r for r in rewards[selected_task_idx]]) / len(
                    [r for r in rewards[selected_task_idx]]
                )
                # st.write(f"**Average Reward:** {average_reward} (error: {rewards[selected_task_idx].count(None)})")
            for idx, element in enumerate(matching_elements, 1):
                key = f"{element['task_idx']}-{element['trial']}"
                with st.expander(
                    f"Sample {idx} for Task {selected_task_idx} (Reward: {rewards[selected_task_idx][idx - 1]})",
                    expanded=False,
                ):
                    task_info = element["info"]["task"]
                    reward = element["reward"]
                    task_info = element["info"]["task"]
                    reward_info = element["info"]["reward_info"]
                    messages = element["messages"]

                    st.markdown("**Instruction:**")
                    st.write(task_info["instruction"])

                    if task_info["gold_answer"] is not None:
                        st.markdown("**Gold SQL:**")
                        pretty_sql = sqlparse.format(
                            task_info["gold_sql"], reindent=True, keyword_case="upper"
                        )
                        st.code(pretty_sql, language="sql")

                        st.markdown("**Gold Answer:**")
                        gold_answer = [l[0] for l in task_info["gold_answer"]]
                        st.code(gold_answer)

                    st.markdown("---")

                    st.subheader("Conversation")

                    if task_info["gold_answer"] is not None:
                        st.markdown(f"**Reward:** {reward}")
                        if reward_info["reward"]:
                            st.code(f"{reward_info['info']['pred_sql']}")
                            st.code(f"{reward_info['info']['pred_answer']}")
                        else:
                            st.code("N/A")
                            st.code("N/A")

                    st.markdown("---")

                    for message in messages:
                        role = message.get("role", "user")
                        if role == "system":
                            continue
                        content = message.get("content")

                        if role == "user":
                            with st.chat_message("user"):
                                st.markdown(content)
                        elif role == "assistant":
                            if content:
                                with st.chat_message("assistant"):
                                    st.markdown(content)
                            if message.get("tool_calls", []):
                                for call in message.get("tool_calls", []):
                                    func = call.get("function", {})
                                    name = func.get("name", "Unknown tool")
                                    arguments = func.get(
                                        "arguments", "No arguments provided"
                                    )
                                    tool_info_str = f"**Tool used:** {name}\n\n**Arguments:** {arguments}\n\n"
                                    with st.chat_message("assistant"):
                                        st.markdown(tool_info_str)
                        elif role == "tool":
                            with st.chat_message("assistant"):
                                st.markdown("*Tool Response:*")
                                st.code(content, language="sql")
                        else:
                            st.markdown(f"**{role.capitalize()}:** {content}")

                    if element.get("error_traceback", None):
                        st.code(element["error_traceback"])
    else:
        st.json(data, expanded=True)
