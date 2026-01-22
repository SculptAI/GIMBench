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

python -m gimbench.ctp.gim_sft --model_type openai --model_name openai/gpt-5.2 --api_key $API_KEY --base_url $API_BASE --use_gim_prompt --output_type json --ref_model_device cpu --first_n 100 --base_model_vocab_size 201088
python -m gimbench.ctp.gim_sft --model_type openai --model_name google/gemini-3-flash-preview --api_key $API_KEY --base_url $API_BASE --use_gim_prompt --output_type json --ref_model_device cpu --first_n 100 --base_model_vocab_size 256128
python -m gimbench.ctp.gim_sft --model_type openai --model_name moonshotai/kimi-k2-thinking --api_key $API_KEY --base_url $API_BASE --use_gim_prompt --output_type json --ref_model_device cpu --first_n 100 --base_model_vocab_size 163840
python -m gimbench.ctp.gim_sft --model_type openai --model_name deepseek/deepseek-chat-v3-0324 --api_key $API_KEY --base_url $API_BASE --use_gim_prompt --output_type json --ref_model_device cpu --first_n 100 --base_model_vocab_size 129280
python -m gimbench.ctp.gim_sft --model_type vllm-offline --model_name Qwen/Qwen3-4B-Instruct-2507 --use_gim_prompt --output_type cfg --ref_model_device cpu --first_n 100 --base_model_vocab_size 151936
python -m gimbench.ctp.gim_sft --model_type vllm-offline --model_name Qwen/Qwen3-1.7B --use_gim_prompt --output_type cfg --ref_model_device cpu --first_n 100 --base_model_vocab_size 151936
python -m gimbench.ctp.gim_sft --model_type vllm-offline --model_name Sculpt-AI/09251-gim-sft-tmp --output_type cfg --ref_model_device cpu --first_n 100 --base_model_vocab_size 151936
python -m gimbench.ctp.gim_sft --model_type vllm-offline --model_name Sculpt-AI/2510231-mixture-high --output_type cfg --ref_model_device cpu --first_n 100 --base_model_vocab_size 151936
python -m gimbench.ctp.gim_sft --model_type vllm-offline --model_name Sculpt-AI/2510232-mixture-mid --output_type cfg --ref_model_device cpu --first_n 100 --base_model_vocab_size 151936
python -m gimbench.ctp.gim_sft --model_type vllm-offline --model_name Sculpt-AI/2601081-mix1 --output_type cfg --ref_model_device cpu --first_n 100 --base_model_vocab_size 151936
python -m gimbench.ctp.gim_sft --model_type vllm-offline --model_name Sculpt-AI/2601091-mix2 --output_type cfg --ref_model_device cpu --first_n 100 --base_model_vocab_size 151936
python -m gimbench.ctp.gim_sft --model_type vllm-offline --model_name Sculpt-AI/2601092-mix3 --output_type cfg --ref_model_device cpu --first_n 100 --base_model_vocab_size 151936
python -m gimbench.ctp.gim_sft --model_type vllm-offline --model_name Sculpt-AI/2601093-mix4 --output_type cfg --ref_model_device cpu --first_n 100 --base_model_vocab_size 151936

python -m gimbench.match.gim_regex --model_type openai --model_name openai/gpt-5.2 --api_key $API_KEY --base_url $API_BASE --use_gim_prompt --output_type json --first_n 100
python -m gimbench.match.gim_regex --model_type openai --model_name google/gemini-3-flash-preview --api_key $API_KEY --base_url $API_BASE --use_gim_prompt --output_type json --first_n 100
python -m gimbench.match.gim_regex --model_type openai --model_name moonshotai/kimi-k2-thinking --api_key $API_KEY --base_url $API_BASE --use_gim_prompt --output_type json --first_n 100
python -m gimbench.match.gim_regex --model_type openai --model_name deepseek/deepseek-chat-v3-0324 --api_key $API_KEY --base_url $API_BASE --use_gim_prompt --output_type json --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Qwen/Qwen3-4B-Instruct-2507 --use_gim_prompt --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Qwen/Qwen3-1.7B --use_gim_prompt --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/09251-gim-sft-tmp --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2510231-mixture-high --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2510232-mixture-mid --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2601081-mix1 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2601091-mix2 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2601092-mix3 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2601093-mix4 --output_type cfg --ref_model_device cpu --first_n 100

shutdown -h +3
