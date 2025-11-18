# DO NOT RUN THIS SCRIPT DIRECTLY.
# Commands below are intended to be copy-pasted into a terminal.

# Count of examples in each dataset split:
# #(Sculpt-AI/GIM-SFT/*/train) = 3157829
# #(Idavidrein/gpqa/gpqa_diamond/train) = 198
# #(openlifescienceai/medmcqa/validation) = 4183
# #(TIGER-Lab/MMLU-Pro/test) = 12102
# #(allenai/qasc/validation) = 920

make serve model_path=artifacts/09251-gim-sft-tmp/sft-gim
python -m eval.ctp.gim_sft --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --ref_model_device cpu --first_n 100
for BUDGET in {2..18..2}; do python -m eval.mcqa.gpqa_diamond --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 198; done
for BUDGET in {2..18..2}; do python -m eval.mcqa.medmcqa --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 1000; done
for BUDGET in {2..18..2}; do python -m eval.mcqa.mmlu_pro --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 1000; done
for BUDGET in {2..18..2}; do python -m eval.mcqa.qasc --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 920; done

make serve model_path=unsloth/Qwen3-4B-Instruct-2507
python -m eval.mcqa.gpqa_diamond --model_name unsloth/Qwen3-4B-Instruct-2507 --num_proc 40 --first_n 198
python -m eval.mcqa.medmcqa --model_name unsloth/Qwen3-4B-Instruct-2507 --num_proc 40 --first_n 1000
python -m eval.mcqa.mmlu_pro --model_name unsloth/Qwen3-4B-Instruct-2507 --num_proc 40 --first_n 1000
python -m eval.mcqa.qasc --model_name unsloth/Qwen3-4B-Instruct-2507 --num_proc 40 --first_n 920

make serve model_path=artifacts/2510231-mixture-high/sft-gim
python -m eval.ctp.gim_sft --model_name artifacts/2510231-mixture-high/sft-gim --is_gim --ref_model_device cpu --first_n 100
for BUDGET in {2..18..2}; do python -m eval.mcqa.gpqa_diamond --model_name artifacts/2510231-mixture-high/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 198; done
for BUDGET in {2..18..2}; do python -m eval.mcqa.medmcqa --model_name artifacts/2510231-mixture-high/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 1000; done
for BUDGET in {2..18..2}; do python -m eval.mcqa.mmlu_pro --model_name artifacts/2510231-mixture-high/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 1000; done
for BUDGET in {2..18..2}; do python -m eval.mcqa.qasc --model_name artifacts/2510231-mixture-high/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 920; done
sudo shutdown -h +3

make serve model_path=artifacts/2510232-mixture-mid/sft-gim
python -m eval.ctp.gim_sft --model_name artifacts/2510232-mixture-mid/sft-gim --is_gim --ref_model_device cpu --first_n 100
for BUDGET in {2..18..2}; do python -m eval.mcqa.gpqa_diamond --model_name artifacts/2510232-mixture-mid/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 198; done
for BUDGET in {2..18..2}; do python -m eval.mcqa.medmcqa --model_name artifacts/2510232-mixture-mid/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 1000; done
for BUDGET in {2..18..2}; do python -m eval.mcqa.mmlu_pro --model_name artifacts/2510232-mixture-mid/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 1000; done
for BUDGET in {2..18..2}; do python -m eval.mcqa.qasc --model_name artifacts/2510232-mixture-mid/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 920; done
sudo shutdown -h +3

make serve model_path=artifacts/2510233-mixture-low/sft-gim
python -m eval.ctp.gim_sft --model_name artifacts/2510233-mixture-low/sft-gim --is_gim --ref_model_device cpu --first_n 100
for BUDGET in {2..18..2}; do python -m eval.mcqa.gpqa_diamond --model_name artifacts/2510233-mixture-low/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 198; done
for BUDGET in {2..18..2}; do python -m eval.mcqa.medmcqa --model_name artifacts/2510233-mixture-low/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 1000; done
for BUDGET in {2..18..2}; do python -m eval.mcqa.mmlu_pro --model_name artifacts/2510233-mixture-low/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 1000; done
for BUDGET in {2..18..2}; do python -m eval.mcqa.qasc --model_name artifacts/2510233-mixture-low/sft-gim --is_gim --reason_budget "$BUDGET" --num_proc 40 --first_n 920; done
sudo shutdown -h +3
