export API_KEY=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
export API_BASE=https://openrouter.ai/api/v1

python -m gimbench.cv.cv_parse --use_uie --model_name "PP-UIE-7B" --api_key $API_KEY --base_url $API_BASE

shutdown -h +3
