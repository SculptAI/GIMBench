#!/bin/bash

set -x

export API_KEY=xxxxxx
export API_BASE=https://openrouter.ai/api/v1

local_gim_models=(
    "Sculpt-AI/GIM-1.7B"
    "Sculpt-AI/GIM-4B"
)
for model in "${local_gim_models[@]}"; do
    python -m gimbench.cv.cv_parse --model_type vllm-offline --model_name $model --output_type cfg --api_key $API_KEY --base_url $API_BASE
done

local_non_gim_models=(
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B-Instruct-2507"
)
for model in "${local_non_gim_models[@]}"; do
    python -m gimbench.cv.cv_parse --model_type vllm-offline --model_name $model --use_gim_prompt --output_type cfg --api_key $API_KEY --base_url $API_BASE
    python -m gimbench.cv.cv_parse --model_type vllm-offline --model_name $model --use_outlines --api_key $API_KEY --base_url $API_BASE
done

local_gim_models=(
    "Sculpt-AI/GIM-1.7B"
    "Sculpt-AI/GIM-4B"
)
for model in "${local_gim_models[@]}"; do
    python -m gimbench.cv.cv_parse --model_type vllm-offline --model_name $model --output_type cfg --api_key $API_KEY --base_url $API_BASE --presence_penalty 0
done

local_non_gim_models=(
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B-Instruct-2507"
)
for model in "${local_non_gim_models[@]}"; do
    python -m gimbench.cv.cv_parse --model_type vllm-offline --model_name $model --use_gim_prompt --output_type cfg --api_key $API_KEY --base_url $API_BASE --presence_penalty 0
    python -m gimbench.cv.cv_parse --model_type vllm-offline --model_name $model --use_outlines --api_key $API_KEY --base_url $API_BASE --presence_penalty 0
done

shutdown -h +3
