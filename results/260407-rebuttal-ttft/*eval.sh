#!/bin/bash

set -x

python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Qwen/Qwen3-4B-Instruct-2507 --use_gim_prompt --output_type none --ref_model_device cpu --first_n 100 --record_timing
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Qwen/Qwen3-1.7B --use_gim_prompt --output_type none --ref_model_device cpu --first_n 100 --record_timing
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Qwen/Qwen3-4B-Instruct-2507 --use_gim_prompt --output_type json --ref_model_device cpu --first_n 100 --record_timing
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Qwen/Qwen3-1.7B --use_gim_prompt --output_type json --ref_model_device cpu --first_n 100 --record_timing
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Qwen/Qwen3-4B-Instruct-2507 --use_gim_prompt --output_type cfg --ref_model_device cpu --first_n 100 --record_timing
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Qwen/Qwen3-1.7B --use_gim_prompt --output_type cfg --ref_model_device cpu --first_n 100 --record_timing
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Sculpt-AI/GIM-4B --output_type json --ref_model_device cpu --first_n 100 --record_timing
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Sculpt-AI/GIM-1.7B --output_type json --ref_model_device cpu --first_n 100 --record_timing
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Sculpt-AI/GIM-4B --output_type cfg --ref_model_device cpu --first_n 100 --record_timing
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Sculpt-AI/GIM-1.7B --output_type cfg --ref_model_device cpu --first_n 100 --record_timing

python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Qwen/Qwen3-4B-Instruct-2507 --use_gim_prompt --output_type none --ref_model_device cpu --first_n 100 --record_timing
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Qwen/Qwen3-1.7B --use_gim_prompt --output_type none --ref_model_device cpu --first_n 100 --record_timing
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Qwen/Qwen3-4B-Instruct-2507 --use_gim_prompt --output_type json --ref_model_device cpu --first_n 100 --record_timing
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Qwen/Qwen3-1.7B --use_gim_prompt --output_type json --ref_model_device cpu --first_n 100 --record_timing
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Qwen/Qwen3-4B-Instruct-2507 --use_gim_prompt --output_type cfg --ref_model_device cpu --first_n 100 --record_timing
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Qwen/Qwen3-1.7B --use_gim_prompt --output_type cfg --ref_model_device cpu --first_n 100 --record_timing
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/GIM-4B --output_type json --ref_model_device cpu --first_n 100 --record_timing
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/GIM-1.7B --output_type json --ref_model_device cpu --first_n 100 --record_timing
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/GIM-4B --output_type cfg --ref_model_device cpu --first_n 100 --record_timing
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/GIM-1.7B --output_type cfg --ref_model_device cpu --first_n 100 --record_timing

shutdown -h +3
