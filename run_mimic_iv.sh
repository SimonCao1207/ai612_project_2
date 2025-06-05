# # gpt-4o-mini
python run.py --env mimic_iv \
    --user_model gemini/gemini-2.0-flash \
    --model gemini/gemini-2.0-flash \
    --agent_strategy tool-calling \
    --temperature 0.0 \
    --seed 42 \
    --num_trials 4 \
    --max_concurrency 5 \
    --eval_mode valid # "valid" if running on the validation set. "test" if running on the test set.

# TODO: change --agent_strategy to your own implementation
# max_concurrency is the number of concurrent runs of the Tasks in the evaluation data.


# # gemini
# python run.py --env mimic_iv \
#     --user_model gemini/gemini-2.0-flash \
#     --model gemini/gemini-2.0-flash \
#     --agent_strategy tool-calling-v2 \
#     --temperature 0.0 \
#     --seed 42 \
#     --num_trials 4 \
#     --max_concurrency 1 \
#     --eval_mode valid
#     --task_ids 5


# python run.py --env mimic_iv \
#     --model gpt-4o-mini \
#     --agent_strategy tool-calling \
#     --temperature 0.0 \
#     --seed 42 \
#     --num_trials 1 \
#     --max_concurrency 1 \
#     --eval_mode valid \
#     --task_ids 3 6

