#!/bin/bash

set -x

python -m gimbench.code.humaneval_infilling --model_type vllm-offline --model_name Qwen/Qwen3-1.7B --no_gimkit --first_n 100
python -m gimbench.code.humaneval_infilling --model_type vllm-offline --model_name Sculpt-AI/GIM-1.7B --output_type cfg --first_n 100

python -m gimbench.code.humaneval_infilling --model_type vllm-offline --model_name Qwen/Qwen3-4B-Instruct-2507 --no_gimkit --first_n 100
python -m gimbench.code.humaneval_infilling --model_type vllm-offline --model_name Sculpt-AI/GIM-4B --output_type cfg --first_n 100

shutdown -h +3
