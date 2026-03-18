#!/bin/bash

set -x

$API_KEY=xxx
$API_BASE=xxx
$MODEL=Sculpt-AI/2603171-gemma

hf download $MODEL

python -m gimbench.ppl.gim_sft --ref_model_device cpu --golden_truth_only --first_n 100
python -m gimbench.ppl.gim_sft --model_type vllm-offline --model_name $MODEL --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name $MODEL --output_type cfg --ref_model_device cpu --first_n 100

python -m "gimbench.mcqa.gpqa_diamond" --model_type vllm-offline --model_name "$MODEL" --auto_budget --num_proc 40 --first_n 198
python -m "gimbench.mcqa.medmcqa" --model_type vllm-offline --model_name "$MODEL" --auto_budget --num_proc 40 --first_n 1000
python -m "gimbench.mcqa.mmlu_pro" --model_type vllm-offline --model_name "$MODEL" --auto_budget --num_proc 40 --first_n 1000
python -m "gimbench.mcqa.qasc" --model_type vllm-offline --model_name "$MODEL" --auto_budget --num_proc 40 --first_n 920

python -m "gimbench.mcqa.gpqa_diamond" --model_type vllm-offline --model_name "google/gemma-3-270m-it" --no_gimkit --num_proc 40 --first_n 198
python -m "gimbench.mcqa.medmcqa" --model_type vllm-offline --model_name "google/gemma-3-270m-it" --no_gimkit --num_proc 40 --first_n 1000
python -m "gimbench.mcqa.mmlu_pro" --model_type vllm-offline --model_name "google/gemma-3-270m-it" --no_gimkit --num_proc 40 --first_n 1000
python -m "gimbench.mcqa.qasc" --model_type vllm-offline --model_name "google/gemma-3-270m-it" --no_gimkit --num_proc 40 --first_n 920

python -m gimbench.cv.cv_parse --model_type vllm-offline --model_name $MODEL --output_type cfg --api_key $API_KEY --base_url $API_BASE

shutdown -h +3
