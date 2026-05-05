#!/bin/bash

set -x

API_KEY="$1"
API_BASE="https://openrouter.ai/api/v1"

hf download google/gemma-4-E2B-it
hf download google/gemma-3-1b-it
hf download google/gemma-3-270m-it
hf download Qwen/Qwen3.5-4B
hf download Qwen/Qwen3.5-2B
hf download Qwen/Qwen3.5-0.8B

MODEL_NAME="/mnt/data/artifacts/2604261-gemma-e2b-update/sft-gim"

python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name "$MODEL_NAME" --output_type cfg --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name "$MODEL_NAME" --output_type cfg --first_n 100
python -m gimbench.code.humaneval_infilling --model_type vllm-offline --model_name "$MODEL_NAME" --output_type cfg --first_n 200
python -m gimbench.cv.cv_parse --model_type vllm-offline --model_name "$MODEL_NAME" --output_type cfg --api_key "$API_KEY" --base_url "$API_BASE"
python -m gimbench.scierc.scierc --model_type vllm-offline --model_name "$MODEL_NAME" --output_type cfg

# python -m gimbench.mcqa.gpqa_diamond --model_type vllm-offline --model_name "$MODEL_NAME" --auto_budget --first_n 100
# python -m gimbench.mcqa.medmcqa --model_type vllm-offline --model_name "$MODEL_NAME" --auto_budget --first_n 100
# python -m gimbench.mcqa.mmlu_pro --model_type vllm-offline --model_name "$MODEL_NAME" --auto_budget --first_n 100
# python -m gimbench.mcqa.qasc --model_type vllm-offline --model_name "$MODEL_NAME" --auto_budget --first_n 100



BASE_MODEL="google/gemma-4-E2B-it"

python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name "$BASE_MODEL" --use_gim_prompt --output_type cfg --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name "$BASE_MODEL" --use_gim_prompt --output_type cfg --first_n 100
python -m gimbench.code.humaneval_infilling --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 200
python -m gimbench.cv.cv_parse --model_type vllm-offline --model_name "$BASE_MODEL" --use_outlines --api_key "$API_KEY" --base_url "$API_BASE"
python -m gimbench.scierc.scierc --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit

# python -m gimbench.mcqa.gpqa_diamond --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 100
# python -m gimbench.mcqa.medmcqa --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 100
# python -m gimbench.mcqa.mmlu_pro --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 100
# python -m gimbench.mcqa.qasc --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 100



BASE_MODEL="google/gemma-3-1b-it"

python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name "$BASE_MODEL" --use_gim_prompt --output_type cfg --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name "$BASE_MODEL" --use_gim_prompt --output_type cfg --first_n 100
python -m gimbench.code.humaneval_infilling --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 200
python -m gimbench.cv.cv_parse --model_type vllm-offline --model_name "$BASE_MODEL" --use_outlines --api_key "$API_KEY" --base_url "$API_BASE"
python -m gimbench.scierc.scierc --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit

# python -m gimbench.mcqa.gpqa_diamond --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 100
# python -m gimbench.mcqa.medmcqa --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 100
# python -m gimbench.mcqa.mmlu_pro --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 100
# python -m gimbench.mcqa.qasc --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 100



BASE_MODEL="google/gemma-3-270m-it"

python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name "$BASE_MODEL" --use_gim_prompt --output_type cfg --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name "$BASE_MODEL" --use_gim_prompt --output_type cfg --first_n 100
python -m gimbench.code.humaneval_infilling --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 200
python -m gimbench.cv.cv_parse --model_type vllm-offline --model_name "$BASE_MODEL" --use_outlines --api_key "$API_KEY" --base_url "$API_BASE"
python -m gimbench.scierc.scierc --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit

# python -m gimbench.mcqa.gpqa_diamond --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 100
# python -m gimbench.mcqa.medmcqa --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 100
# python -m gimbench.mcqa.mmlu_pro --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 100
# python -m gimbench.mcqa.qasc --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 100



BASE_MODEL="Qwen/Qwen3.5-4B"

python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name "$BASE_MODEL" --use_gim_prompt --output_type cfg --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name "$BASE_MODEL" --use_gim_prompt --output_type cfg --first_n 100
python -m gimbench.code.humaneval_infilling --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 200
python -m gimbench.cv.cv_parse --model_type vllm-offline --model_name "$BASE_MODEL" --use_outlines --api_key "$API_KEY" --base_url "$API_BASE"
python -m gimbench.scierc.scierc --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit

# python -m gimbench.mcqa.gpqa_diamond --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 100
# python -m gimbench.mcqa.medmcqa --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 100
# python -m gimbench.mcqa.mmlu_pro --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 100
# python -m gimbench.mcqa.qasc --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 100



BASE_MODEL="Qwen/Qwen3.5-2B"

python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name "$BASE_MODEL" --use_gim_prompt --output_type cfg --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name "$BASE_MODEL" --use_gim_prompt --output_type cfg --first_n 100
python -m gimbench.code.humaneval_infilling --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 200
python -m gimbench.cv.cv_parse --model_type vllm-offline --model_name "$BASE_MODEL" --use_outlines --api_key "$API_KEY" --base_url "$API_BASE"
python -m gimbench.scierc.scierc --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit

# python -m gimbench.mcqa.gpqa_diamond --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 100
# python -m gimbench.mcqa.medmcqa --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 100
# python -m gimbench.mcqa.mmlu_pro --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 100
# python -m gimbench.mcqa.qasc --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 100



BASE_MODEL="Qwen/Qwen3.5-0.8B"

python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name "$BASE_MODEL" --use_gim_prompt --output_type cfg --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name "$BASE_MODEL" --use_gim_prompt --output_type cfg --first_n 100
python -m gimbench.code.humaneval_infilling --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 200
python -m gimbench.cv.cv_parse --model_type vllm-offline --model_name "$BASE_MODEL" --use_outlines --api_key "$API_KEY" --base_url "$API_BASE"
python -m gimbench.scierc.scierc --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit

# python -m gimbench.mcqa.gpqa_diamond --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 100
# python -m gimbench.mcqa.medmcqa --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 100
# python -m gimbench.mcqa.mmlu_pro --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 100
# python -m gimbench.mcqa.qasc --model_type vllm-offline --model_name "$BASE_MODEL" --no_gimkit --first_n 100
