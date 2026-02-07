#!/bin/bash
set -x


GIM_MODELS=(
    "Sculpt-AI/2601261-qwen-4b-50k"
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


run_gim_experiments() {
    for ds in "${DATASETS[@]}"; do
        python -m "gimbench.mcqa.$ds" --model_type vllm-offline --model_name "Qwen/Qwen3-4B-Instruct-2507" \
            --no_gimkit --num_proc 40 --first_n -1
        for model in "${GIM_MODELS[@]}"; do
            python -m "gimbench.mcqa.$ds" --model_type vllm-offline --model_name "$model" \
                --auto_budget --auto_budget_prompt "$AUTO_BUDGET_PROMPT" \
                --reason_step_desc "$REASON_STEP_DESC" --num_proc 40 --first_n -1
            for BUDGET in {1..7..2}; do python -m "gimbench.mcqa.$ds" --model_type vllm-offline --model_name "$model" \
                 --reason_budget "$BUDGET" --num_proc 40 --first_n -1; done
        done
    done
}



setup_prompt
run_gim_experiments



shutdown -h +3
