set -x

cd /root/autodl-tmp/GIMBench

MODELS=(
"/root/autodl-tmp/GIM-research/artifacts/2601081-mix1/sft-gim"
"/root/autodl-tmp/GIM-research/artifacts/2601091-mix2/sft-gim"
"/root/autodl-tmp/GIM-research/artifacts/2601092-mix3/sft-gim"
"/root/autodl-tmp/GIM-research/artifacts/2601093-mix4/sft-gim"
)

for MODEL_NAME in "${MODELS[@]}"; do
    python -m gimbench.ctp.gim_sft --model_type vllm-offline --model_name $MODEL_NAME --ref_model_device cpu --first_n 100
    for BUDGET in {2..18..2}; do python -m gimbench.mcqa.gpqa_diamond --model_type vllm-offline --model_name $MODEL_NAME --reason_budget "$BUDGET" --num_proc 40 --first_n 198; done
    for BUDGET in {2..18..2}; do python -m gimbench.mcqa.medmcqa --model_type vllm-offline --model_name $MODEL_NAME --reason_budget "$BUDGET" --num_proc 40 --first_n 1000; done
    for BUDGET in {2..18..2}; do python -m gimbench.mcqa.mmlu_pro --model_type vllm-offline --model_name $MODEL_NAME --reason_budget "$BUDGET" --num_proc 40 --first_n 1000; done
    for BUDGET in {2..18..2}; do python -m gimbench.mcqa.qasc --model_type vllm-offline --model_name $MODEL_NAME --reason_budget "$BUDGET" --num_proc 40 --first_n 920; done
    python -m gimbench.match.gim_regex --model_type vllm-offline --model_name $MODEL_NAME --first_n 100 --output_type cfg
done

sudo shutdown -h +3
