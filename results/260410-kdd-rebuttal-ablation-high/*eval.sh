#!/bin/bash

set -x

export API_KEY=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
export API_BASE=https://openrouter.ai/api/v1

# hf download "Sculpt-AI/2604071-rebuttal-long-context"

local_gim_models=(
    "/root/autodl-tmp/GIM-research/artifacts/2604101-rebuttal-ablation-high/sft-gim"
)
for model in "${local_gim_models[@]}"; do
    python -m gimbench.cv.cv_parse --model_type vllm-offline --model_name $model --output_type cfg --api_key $API_KEY --base_url $API_BASE
done

# shutdown -h +3
