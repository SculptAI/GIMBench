#!/bin/bash
set -x


GIM_MODELS=(
    "Sculpt-AI/2601261-qwen-4b"
    "Sculpt-AI/2601261-qwen-4b-50k"
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
    python -m "gimbench.mcqa.medmcqa" --model_type vllm-offline --model_name "Qwen/Qwen3-4B-Instruct-2507" \
    --no_gimkit --num_proc 40 --first_n 500

    python -m "gimbench.mcqa.medmcqa" --use_gim_prompt --output_type cfg --model_type vllm-offline --model_name "Qwen/Qwen3-4B-Instruct-2507" \
        --auto_budget --auto_budget_prompt "$AUTO_BUDGET_PROMPT" \
        --reason_step_desc "$REASON_STEP_DESC" --num_proc 40 --first_n 500
    for BUDGET in {1..7..2}; do python -m "gimbench.mcqa.medmcqa" --use_gim_prompt --output_type cfg --model_type vllm-offline --model_name "Qwen/Qwen3-4B-Instruct-2507" \
        --reason_budget "$BUDGET" --num_proc 40 --first_n 500; done

    python -m "gimbench.mcqa.medmcqa" --use_gim_prompt --output_type cfg --model_type vllm-offline --model_name "Sculpt-AI/2601261-qwen-4b-50k" \
        --auto_budget --auto_budget_prompt "$AUTO_BUDGET_PROMPT" \
        --reason_step_desc "$REASON_STEP_DESC" --num_proc 40 --first_n 500
    for BUDGET in {1..7..2}; do python -m "gimbench.mcqa.medmcqa" --use_gim_prompt --output_type cfg --model_type vllm-offline --model_name "Sculpt-AI/2601261-qwen-4b-50k" \
        --reason_budget "$BUDGET" --num_proc 40 --first_n 500; done

    python -m "gimbench.mcqa.qasc" --use_gim_prompt --output_type cfg --model_type vllm-offline --model_name "Qwen/Qwen3-4B-Instruct-2507" \
        --auto_budget --auto_budget_prompt "$AUTO_BUDGET_PROMPT" \
        --reason_step_desc "$REASON_STEP_DESC" --num_proc 40 --first_n -1
    for BUDGET in {1..7..2}; do python -m "gimbench.mcqa.qasc" --use_gim_prompt --output_type cfg --model_type vllm-offline --model_name "Qwen/Qwen3-4B-Instruct-2507" \
        --reason_budget "$BUDGET" --num_proc 40 --first_n -1; done

    python -m "gimbench.mcqa.qasc" --use_gim_prompt --output_type cfg --model_type vllm-offline --model_name "Sculpt-AI/2601261-qwen-4b-50k" \
        --auto_budget --auto_budget_prompt "$AUTO_BUDGET_PROMPT" \
        --reason_step_desc "$REASON_STEP_DESC" --num_proc 40 --first_n -1
    for BUDGET in {1..7..2}; do python -m "gimbench.mcqa.qasc" --use_gim_prompt --output_type cfg --model_type vllm-offline --model_name "Sculpt-AI/2601261-qwen-4b-50k" \
        --reason_budget "$BUDGET" --num_proc 40 --first_n -1; done

}



setup_prompt
run_gim_experiments



shutdown -h +3
