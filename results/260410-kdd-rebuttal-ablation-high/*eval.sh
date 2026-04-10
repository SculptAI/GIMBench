#!/bin/bash

set -x

export API_KEY=xxx
export API_BASE=https://openrouter.ai/api/v1


local_gim_models=(
    "/root/autodl-tmp/GIM-research/artifacts/2604101-rebuttal-ablation-high/sft-gim"
)
for model in "${local_gim_models[@]}"; do
    python -m gimbench.cv.cv_parse --model_type vllm-offline --model_name $model --output_type cfg --api_key $API_KEY --base_url $API_BASE
done
