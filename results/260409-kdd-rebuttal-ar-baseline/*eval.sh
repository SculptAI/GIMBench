#!/bin/bash

set -x

hf download Sculpt-AI/2604081-ar-baseline


python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Sculpt-AI/2604081-ar-baseline --use_gim_prompt --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name Sculpt-AI/2604081-ar-baseline --use_gim_prompt --output_type json --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2604081-ar-baseline --use_gim_prompt --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2604081-ar-baseline --use_gim_prompt --output_type json --ref_model_device cpu --first_n 100

shutdown -h +3
