#!/bin/bash

set -x

API_KEY="$1"
API_BASE="https://openrouter.ai/api/v1"
JUDGE_MODEL="google/gemini-3-flash-preview"


MODEL_NAME="/mnt/data/artifacts/2606081-gim4b-rl/rlvr-gim-model/"

# python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name "$MODEL_NAME" --output_type cfg --first_n 100
# python -m gimbench.match.gim_regex --model_type vllm-offline --model_name "$MODEL_NAME" --output_type cfg --first_n 100
# python -m gimbench.code.humaneval_infilling --model_type vllm-offline --model_name "$MODEL_NAME" --output_type cfg --first_n 200
# python -m gimbench.cv.cv_parse --model_type vllm-offline --model_name "$MODEL_NAME" --output_type cfg --api_key "$API_KEY" --base_url "$API_BASE" --judge_model_name "$JUDGE_MODEL"
# python -m gimbench.scierc.scierc --model_type vllm-offline --model_name "$MODEL_NAME" --output_type cfg

# python -m gimbench.mcqa.gpqa_diamond --model_type vllm-offline --model_name "$MODEL_NAME" --auto_budget --first_n 100
# python -m gimbench.mcqa.medmcqa --model_type vllm-offline --model_name "$MODEL_NAME" --auto_budget --first_n 100
# python -m gimbench.mcqa.mmlu_pro --model_type vllm-offline --model_name "$MODEL_NAME" --auto_budget --first_n 100
# python -m gimbench.mcqa.qasc --model_type vllm-offline --model_name "$MODEL_NAME" --auto_budget --first_n 100


MODEL_NAME="Sculpt-AI/GIM-4B"

# python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name "$MODEL_NAME" --output_type cfg --first_n 100
# python -m gimbench.match.gim_regex --model_type vllm-offline --model_name "$MODEL_NAME" --output_type cfg --first_n 100
# python -m gimbench.code.humaneval_infilling --model_type vllm-offline --model_name "$MODEL_NAME" --output_type cfg --first_n 200
# python -m gimbench.cv.cv_parse --model_type vllm-offline --model_name "$MODEL_NAME" --output_type cfg --api_key "$API_KEY" --base_url "$API_BASE" --judge_model_name "$JUDGE_MODEL"
# python -m gimbench.scierc.scierc --model_type vllm-offline --model_name "$MODEL_NAME" --output_type cfg

# python -m gimbench.mcqa.gpqa_diamond --model_type vllm-offline --model_name "$MODEL_NAME" --auto_budget --first_n 100
# python -m gimbench.mcqa.medmcqa --model_type vllm-offline --model_name "$MODEL_NAME" --auto_budget --first_n 100
# python -m gimbench.mcqa.mmlu_pro --model_type vllm-offline --model_name "$MODEL_NAME" --auto_budget --first_n 100
# python -m gimbench.mcqa.qasc --model_type vllm-offline --model_name "$MODEL_NAME" --auto_budget --first_n 100


MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"

# python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name "$MODEL_NAME" --use_gim_prompt --output_type cfg --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name "$MODEL_NAME" --use_gim_prompt --output_type cfg --first_n 100
python -m gimbench.code.humaneval_infilling --model_type vllm-offline --model_name "$MODEL_NAME" --no_gimkit --output_type cfg --first_n 200
python -m gimbench.cv.cv_parse --model_type vllm-offline --model_name "$MODEL_NAME" --use_outlines --output_type cfg --api_key "$API_KEY" --base_url "$API_BASE" --judge_model_name "$JUDGE_MODEL"
python -m gimbench.scierc.scierc --model_type vllm-offline --model_name "$MODEL_NAME" --use_gim_prompt --output_type cfg

python -m gimbench.mcqa.gpqa_diamond --model_type vllm-offline --model_name "$MODEL_NAME" --no_gimkit --first_n 100
python -m gimbench.mcqa.medmcqa --model_type vllm-offline --model_name "$MODEL_NAME" --no_gimkit --first_n 100
python -m gimbench.mcqa.mmlu_pro --model_type vllm-offline --model_name "$MODEL_NAME" --no_gimkit --first_n 100
python -m gimbench.mcqa.qasc --model_type vllm-offline --model_name "$MODEL_NAME" --no_gimkit --first_n 100
