#!/bin/bash
set -x

export API_KEY=your_api_key_here
export API_BASE="https://openrouter.ai/api/v1"

API_MODELS=(
    "google/gemma-3-27b-it"
    "qwen/qwen3-30b-a3b-instruct-2507"
)

API_MODELS_2=(
    "google/gemma-3-12b-it"
    "xiaomi/mimo-v2-flash"
)


setup_prompt() {
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -f "$script_dir/auto_budget_prompt.txt" ]; then
        export AUTO_BUDGET_PROMPT="$(cat "$script_dir/auto_budget_prompt.txt")"
    else
        echo "Warning: auto_budget_prompt.txt not found in $script_dir" >&2
        export AUTO_BUDGET_PROMPT=""
    fi
    export REASON_STEP_DESC="A distinct, verified reasoning step building logically on the previous one. Each step must be a high-density analysis (180–240 words equivalent) but is fully liberated in format: use frequent line breaks, LaTeX, lists, or tables as needed. The goal is logical transparency; ensure each step achieves a sub-goal, shows its full derivation, and includes a micro-verification to prevent error propagation."
}



run_api_experiments() {
    for model in "${API_MODELS[@]}"; do
        python -m "gimbench.mcqa.medmcqa" --model_type openai --model_name "$model" \
            --api_key "$API_KEY" --base_url "$API_BASE" --no_gimkit --num_proc 40 --first_n 500
        python -m "gimbench.mcqa.medmcqa" --use_gim_prompt --output_type json --model_type openai \
            --model_name "$model" --api_key "$API_KEY" --base_url "$API_BASE" \
            --auto_budget --auto_budget_prompt "$AUTO_BUDGET_PROMPT" \
            --reason_step_desc "$REASON_STEP_DESC" --num_proc 40 --first_n 500
    done
    for model in "${API_MODELS_2[@]}"; do
        python -m "gimbench.mcqa.qasc" --model_type openai --model_name "$model" \
            --api_key "$API_KEY" --base_url "$API_BASE" --no_gimkit --num_proc 40 --first_n -1
        python -m "gimbench.mcqa.qasc" --use_gim_prompt --output_type json --model_type openai \
            --model_name "$model" --api_key "$API_KEY" --base_url "$API_BASE" \
            --auto_budget --auto_budget_prompt "$AUTO_BUDGET_PROMPT" \
            --reason_step_desc "$REASON_STEP_DESC" --num_proc 40 --first_n -1

        python -m "gimbench.mcqa.medmcqa" --model_type openai --model_name "$model" \
            --api_key "$API_KEY" --base_url "$API_BASE" --no_gimkit --num_proc 40 --first_n 500
        python -m "gimbench.mcqa.medmcqa" --use_gim_prompt --output_type json --model_type openai \
            --model_name "$model" --api_key "$API_KEY" --base_url "$API_BASE" \
            --auto_budget --auto_budget_prompt "$AUTO_BUDGET_PROMPT" \
            --reason_step_desc "$REASON_STEP_DESC" --num_proc 40 --first_n 500
    done

    for model in "${API_MODELS[@]}"; do
        for BUDGET in {1..7..2}; do python -m "gimbench.mcqa.medmcqa" --use_gim_prompt --output_type json --model_type openai \
            --model_name "$model" --api_key "$API_KEY" --base_url "$API_BASE" \
            --reason_budget "$BUDGET" --num_proc 40 --first_n 500; done
    done
    for model in "${API_MODELS_2[@]}"; do
        for BUDGET in {1..7..2}; python -m "gimbench.mcqa.qasc" --use_gim_prompt --output_type json --model_type openai \
            --model_name "$model" --api_key "$API_KEY" --base_url "$API_BASE" \
            --reason_budget "$BUDGET" --num_proc 40 --first_n -1; done

        for BUDGET in {1..7..2}; python -m "gimbench.mcqa.medmcqa" --use_gim_prompt --output_type json --model_type openai \
            --model_name "$model" --api_key "$API_KEY" --base_url "$API_BASE" \
            --reason_budget "$BUDGET" --num_proc 40 --first_n 500; done
    done
}



setup_prompt
run_api_experiments

shutdown -h +3
