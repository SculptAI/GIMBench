# Run some exports before running the commands
# export OPENAI_API_KEY="xxx"
# export OPENAI_API_BASE="https://xxx/v1"
# export HF_TOKEN="xxx"

python -m evals.run --model_type openai --api_key $OPENAI_API_KEY --api_base $OPENAI_API_BASE --model_name openai/gpt-5
python -m evals.run --model_type openai --api_key $OPENAI_API_KEY --api_base $OPENAI_API_BASE --model_name openai/gpt-5 --use_gim_prompt
python -m evals.run --model_type openai --api_key $OPENAI_API_KEY --api_base $OPENAI_API_BASE --model_name openai/gpt-5 --use_gim_prompt --output_type json

python -m evals.run --model_type vllm --model_name Qwen/Qwen3-0.6B --use_gim_prompt --output_type cfg

python -m evals.run --model_type vllm --model_name Sculpt-AI/GIM-test
python -m evals.run --model_type vllm --model_name Sculpt-AI/GIM-test --output_type json
python -m evals.run --model_type vllm --model_name Sculpt-AI/GIM-test --output_type cfg
