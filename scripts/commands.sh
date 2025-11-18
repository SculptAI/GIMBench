# TODO: OUTDATED

# ─── COPIED FROM GIM-research/eval/commands.sh ────────────────────────────────

# DO NOT RUN THIS SCRIPT DIRECTLY.
# Commands below are intended to be copy-pasted into a terminal.

# Count of examples in each dataset split:
# #(Sculpt-AI/GIM-SFT/*/train) = 3157829
# #(Idavidrein/gpqa/gpqa_diamond/train) = 198
# #(openlifescienceai/medmcqa/validation) = 4183
# #(TIGER-Lab/MMLU-Pro/test) = 12102
# #(allenai/qasc/validation) = 920

make serve model_path=artifacts/09251-gim-sft-tmp/sft-gim
python -m gimbench.ctp.gim_sft --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --ref_model_device cpu --first_n 100
for BUDGET in {2..18..2}; do python -m gimbench.mcqa.gpqa_diamond --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 198; done
for BUDGET in {2..18..2}; do python -m gimbench.mcqa.medmcqa --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 1000; done
for BUDGET in {2..18..2}; do python -m gimbench.mcqa.mmlu_pro --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 1000; done
for BUDGET in {2..18..2}; do python -m gimbench.mcqa.qasc --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 920; done

make serve model_path=unsloth/Qwen3-4B-Instruct-2507
python -m gimbench.mcqa.gpqa_diamond --model_name unsloth/Qwen3-4B-Instruct-2507 --num_proc 40 --first_n 198
python -m gimbench.mcqa.medmcqa --model_name unsloth/Qwen3-4B-Instruct-2507 --num_proc 40 --first_n 1000
python -m gimbench.mcqa.mmlu_pro --model_name unsloth/Qwen3-4B-Instruct-2507 --num_proc 40 --first_n 1000
python -m gimbench.mcqa.qasc --model_name unsloth/Qwen3-4B-Instruct-2507 --num_proc 40 --first_n 920

make serve model_path=artifacts/2510231-mixture-high/sft-gim
python -m gimbench.ctp.gim_sft --model_name artifacts/2510231-mixture-high/sft-gim --is_gim --ref_model_device cpu --first_n 100
for BUDGET in {2..18..2}; do python -m gimbench.mcqa.gpqa_diamond --model_name artifacts/2510231-mixture-high/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 198; done
for BUDGET in {2..18..2}; do python -m gimbench.mcqa.medmcqa --model_name artifacts/2510231-mixture-high/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 1000; done
for BUDGET in {2..18..2}; do python -m gimbench.mcqa.mmlu_pro --model_name artifacts/2510231-mixture-high/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 1000; done
for BUDGET in {2..18..2}; do python -m gimbench.mcqa.qasc --model_name artifacts/2510231-mixture-high/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 920; done
sudo shutdown -h +3

make serve model_path=artifacts/2510232-mixture-mid/sft-gim
python -m gimbench.ctp.gim_sft --model_name artifacts/2510232-mixture-mid/sft-gim --is_gim --ref_model_device cpu --first_n 100
for BUDGET in {2..18..2}; do python -m gimbench.mcqa.gpqa_diamond --model_name artifacts/2510232-mixture-mid/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 198; done
for BUDGET in {2..18..2}; do python -m gimbench.mcqa.medmcqa --model_name artifacts/2510232-mixture-mid/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 1000; done
for BUDGET in {2..18..2}; do python -m gimbench.mcqa.mmlu_pro --model_name artifacts/2510232-mixture-mid/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 1000; done
for BUDGET in {2..18..2}; do python -m gimbench.mcqa.qasc --model_name artifacts/2510232-mixture-mid/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 920; done
sudo shutdown -h +3

make serve model_path=artifacts/2510233-mixture-low/sft-gim
python -m gimbench.ctp.gim_sft --model_name artifacts/2510233-mixture-low/sft-gim --is_gim --ref_model_device cpu --first_n 100
for BUDGET in {2..18..2}; do python -m gimbench.mcqa.gpqa_diamond --model_name artifacts/2510233-mixture-low/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 198; done
for BUDGET in {2..18..2}; do python -m gimbench.mcqa.medmcqa --model_name artifacts/2510233-mixture-low/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 1000; done
for BUDGET in {2..18..2}; do python -m gimbench.mcqa.mmlu_pro --model_name artifacts/2510233-mixture-low/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 1000; done
for BUDGET in {2..18..2}; do python -m gimbench.mcqa.qasc --model_name artifacts/2510233-mixture-low/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 920; done
sudo shutdown -h +3


# ─── COPIED FROM GIMKit/evals/commands.sh ─────────────────────────────────────

# Run some exports before running the commands
# export OPENAI_API_KEY="xxx"
# export OPENAI_BASE_URL="https://xxx/v1"
# export HF_TOKEN="xxx"

python -m gimbench.match.gim_regex --model_type openai --api_key $OPENAI_API_KEY --base_url $OPENAI_BASE_URL --model_name openai/gpt-5 --first_n 100 
python -m gimbench.match.gim_regex --model_type openai --api_key $OPENAI_API_KEY --base_url $OPENAI_BASE_URL --model_name openai/gpt-5 --first_n 100 --use_gim_prompt
python -m gimbench.match.gim_regex --model_type openai --api_key $OPENAI_API_KEY --base_url $OPENAI_BASE_URL --model_name openai/gpt-5 --first_n 100 --use_gim_prompt --output_type json

python -m gimbench.match.gim_regex --model_type vllm --model_name Qwen/Qwen3-0.6B --first_n 100 --use_gim_prompt --output_type cfg

python -m gimbench.match.gim_regex --model_type vllm --model_name Sculpt-AI/GIM-test --first_n 100
python -m gimbench.match.gim_regex --model_type vllm --model_name Sculpt-AI/GIM-test --first_n 100 --output_type json
python -m gimbench.match.gim_regex --model_type vllm --model_name Sculpt-AI/GIM-test --first_n 100 --output_type cfg
