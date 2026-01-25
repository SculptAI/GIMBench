#!/bin/bash

set -x

export API_KEY="your_api_key_here"
export API_BASE="https://openrouter.ai/api/v1"

hf download Qwen/Qwen3-4B-Instruct-2507
hf download Qwen/Qwen3-1.7B
hf download Sculpt-AI/09251-gim-sft-tmp
hf download Sculpt-AI/2510231-mixture-high
hf download Sculpt-AI/2510232-mixture-mid
hf download Sculpt-AI/2601081-mix1
hf download Sculpt-AI/2601091-mix2
hf download Sculpt-AI/2601092-mix3
hf download Sculpt-AI/2601093-mix4
hf download Sculpt-AI/2601191-grid-lr2e-5-ga12
hf download Sculpt-AI/2601192-grid-lr2e-5-ga16
hf download Sculpt-AI/2601193-grid-lr2e-5-ga24
hf download Sculpt-AI/2601194-grid-lr2e-4-ga12
hf download Sculpt-AI/2601195-grid-lr2e-4-ga16
hf download Sculpt-AI/2601196-grid-lr2e-4-ga24
hf download Sculpt-AI/2601197-grid-lr2e-6-ga12
hf download Sculpt-AI/2601198-grid-lr2e-6-ga16
hf download Sculpt-AI/2601199-grid-lr2e-6-ga24


python -m gimbench.ppl.gim_sft --model_type openai --model_name openai/gpt-5.2 --api_key $API_KEY --base_url $API_BASE --use_gim_prompt --output_type json --ref_model_device cpu --first_n 100
python -m gimbench.ppl.gim_sft --model_type openai --model_name google/gemini-3-flash-preview --api_key $API_KEY --base_url $API_BASE --use_gim_prompt --output_type json --ref_model_device cpu --first_n 100
python -m gimbench.ppl.gim_sft --model_type openai --model_name moonshotai/kimi-k2-thinking --api_key $API_KEY --base_url $API_BASE --use_gim_prompt --output_type json --ref_model_device cpu --first_n 100
python -m gimbench.ppl.gim_sft --model_type openai --model_name deepseek/deepseek-chat-v3-0324 --api_key $API_KEY --base_url $API_BASE --use_gim_prompt --output_type json --ref_model_device cpu --first_n 100
python -m gimbench.ppl.gim_sft --model_type openai --model_name anthropic/claude-haiku-4.5 --api_key $API_KEY --base_url $API_BASE --use_gim_prompt --output_type json --ref_model_device cpu --first_n 100
python -m gimbench.ppl.gim_sft --model_type openai --model_name x-ai/grok-4-fast --api_key $API_KEY --base_url $API_BASE --use_gim_prompt --output_type json --ref_model_device cpu --first_n 100
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Qwen/Qwen3-4B-Instruct-2507 --use_gim_prompt --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Qwen/Qwen3-1.7B --use_gim_prompt --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Sculpt-AI/09251-gim-sft-tmp --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Sculpt-AI/2510231-mixture-high --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Sculpt-AI/2510232-mixture-mid --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Sculpt-AI/2601081-mix1 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Sculpt-AI/2601091-mix2 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Sculpt-AI/2601092-mix3 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Sculpt-AI/2601093-mix4 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Sculpt-AI/2601191-grid-lr2e-5-ga12 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Sculpt-AI/2601192-grid-lr2e-5-ga16 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Sculpt-AI/2601193-grid-lr2e-5-ga24 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Sculpt-AI/2601194-grid-lr2e-4-ga12 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Sculpt-AI/2601195-grid-lr2e-4-ga16 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Sculpt-AI/2601196-grid-lr2e-4-ga24 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Sculpt-AI/2601197-grid-lr2e-6-ga12 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Sculpt-AI/2601198-grid-lr2e-6-ga16 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Sculpt-AI/2601199-grid-lr2e-6-ga24 --output_type cfg --ref_model_device cpu --first_n 100


python -m gimbench.match.gim_regex --model_type openai --model_name openai/gpt-5.2 --api_key $API_KEY --base_url $API_BASE --use_gim_prompt --output_type json --first_n 100
python -m gimbench.match.gim_regex --model_type openai --model_name google/gemini-3-flash-preview --api_key $API_KEY --base_url $API_BASE --use_gim_prompt --output_type json --first_n 100
python -m gimbench.match.gim_regex --model_type openai --model_name moonshotai/kimi-k2-thinking --api_key $API_KEY --base_url $API_BASE --use_gim_prompt --output_type json --first_n 100
python -m gimbench.match.gim_regex --model_type openai --model_name deepseek/deepseek-chat-v3-0324 --api_key $API_KEY --base_url $API_BASE --use_gim_prompt --output_type json --first_n 100
python -m gimbench.match.gim_regex --model_type openai --model_name anthropic/claude-haiku-4.5 --api_key $API_KEY --base_url $API_BASE --use_gim_prompt --output_type json --first_n 100
python -m gimbench.match.gim_regex --model_type openai --model_name x-ai/grok-4-fast --api_key $API_KEY --base_url $API_BASE --use_gim_prompt --output_type json --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Qwen/Qwen3-4B-Instruct-2507 --use_gim_prompt --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Qwen/Qwen3-1.7B --use_gim_prompt --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/09251-gim-sft-tmp --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2510231-mixture-high --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2510232-mixture-mid --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2601081-mix1 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2601091-mix2 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2601092-mix3 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2601093-mix4 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2601191-grid-lr2e-5-ga12 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2601192-grid-lr2e-5-ga16 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2601193-grid-lr2e-5-ga24 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2601194-grid-lr2e-4-ga12 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2601195-grid-lr2e-4-ga16 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2601196-grid-lr2e-4-ga24 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2601197-grid-lr2e-6-ga12 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2601198-grid-lr2e-6-ga16 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2601199-grid-lr2e-6-ga24 --output_type cfg --ref_model_device cpu --first_n 100

shutdown -h +3
