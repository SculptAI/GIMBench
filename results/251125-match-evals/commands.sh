# export OPENAI_API_KEY="xxx"
# export OPENAI_BASE_URL="https://xxx/v1"
# export HF_TOKEN="xxx"


python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Qwen/Qwen3-0.6B --first_n 1000 --use_gim_prompt --output_type cfg
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/GIM-test --first_n 1000 --output_type cfg
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Qwen/Qwen3-1.7B --first_n 1000 --use_gim_prompt --output_type cfg
python -m gimbench.tools.aggregate_results
sudo shutdown -h +3
