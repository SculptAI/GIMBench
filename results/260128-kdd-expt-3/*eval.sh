#!/bin/bash

set -x

api_models=(
    "google/gemini-3-flash-preview"
    "openai/gpt-5.2"
    "moonshotai/kimi-k2-thinking"
    "deepseek/deepseek-chat-v3-0324"
    "anthropic/claude-haiku-4.5"
    "x-ai/grok-4-fast"
    "mistralai/ministral-14b-2512"
    "minimax/minimax-m2.1"
    "z-ai/glm-4.7-flash"
    "google/gemma-3-12b-it"
    "google/gemma-3-27b-it"
    "xiaomi/mimo-v2-flash"
)
for model in "${api_models[@]}"; do
    python -m gimbench.cv.cv_parse --model_type openai --model_name $model --api_key $API_KEY --base_url $API_BASE --use_gim_prompt --output_type json
    python -m gimbench.cv.cv_parse --model_type openai --model_name $model --api_key $API_KEY --base_url $API_BASE --use_outlines
done


local_non_gim_models=(
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B-Instruct-2507"
)
for model in "${local_non_gim_models[@]}"; do
    python -m gimbench.cv.cv_parse --model_type vllm-offline --model_name $model --use_gim_prompt --output_type cfg
done


local_gim_models=(
    "Sculpt-AI/2510231-mixture-high"
    "Sculpt-AI/2510232-mixture-mid"
    "Sculpt-AI/2601081-mix1"
    "Sculpt-AI/2601091-mix2"
    "Sculpt-AI/2601092-mix3"
    "Sculpt-AI/2601093-mix4"
    "Sculpt-AI/2601191-grid-lr2e-5-ga12"
    "Sculpt-AI/2601192-grid-lr2e-5-ga16"
    "Sculpt-AI/2601193-grid-lr2e-5-ga24"
    "Sculpt-AI/2601194-grid-lr2e-4-ga12"
    "Sculpt-AI/2601195-grid-lr2e-4-ga16"
    "Sculpt-AI/2601196-grid-lr2e-4-ga24"
    "Sculpt-AI/2601197-grid-lr2e-6-ga12"
    "Sculpt-AI/2601198-grid-lr2e-6-ga16"
    "Sculpt-AI/2601199-grid-lr2e-6-ga24"
    "Sculpt-AI/2601261-qwen-4b"
    "Sculpt-AI/2601261-qwen-4b-50k"
)
for model in "${local_gim_models[@]}"; do
    python -m gimbench.cv.cv_parse --model_type vllm-offline --model_name $model --output_type cfg
done

shutdown -h +3
