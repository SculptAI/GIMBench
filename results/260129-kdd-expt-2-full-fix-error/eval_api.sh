#!/bin/bash
set -x

export API_KEY=your_api_key_here
export API_BASE="https://openrouter.ai/api/v1"

API_MODELS=(
    "google/gemma-3-27b-it"
    "qwen/qwen3-30b-a3b-instruct-2507"
)

DATASETS=(
    "qasc"
    "gpqa_diamond"
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
    python -m "gimbench.mcqa.gpqa_diamond" --use_gim_prompt --output_type json --model_type openai \
        --model_name "qwen/qwen3-30b-a3b-instruct-2507" --api_key "$API_KEY" --base_url "$API_BASE" --reason_budget 1 --num_proc 40 --first_n -1
    for BUDGET in {1..7..2}; do python -m "gimbench.mcqa.gpqa_diamond" --use_gim_prompt --output_type json --model_type openai \
        --model_name "google/gemma-3-27b-it" --api_key "$API_KEY" --base_url "$API_BASE" --reason_budget "$BUDGET" --num_proc 40 --first_n -1; done
    
    python -m "gimbench.mcqa.qasc" --use_gim_prompt --output_type json --model_type openai \
        --model_name "minimax/minimax-m2.1" --api_key "$API_KEY" --base_url "$API_BASE" --reason_budget 1 --num_proc 40 --first_n -1     
    python -m "gimbench.mcqa.gpqa_diamond" --use_gim_prompt --output_type json --model_type openai \
        --model_name "minimax/minimax-m2.1" --api_key "$API_KEY" --base_url "$API_BASE" --reason_budget 1 --num_proc 40 --first_n -1
    
}



setup_prompt
run_api_experiments

shutdown -h +3