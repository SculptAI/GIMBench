python -m gimbench.cv.cv_parse \
    --use_gliner2 \
    --model_name "fastino/gliner2-large-v1" \
    --model_type "openai" \
    --judge_model_name "google/gemini-2.5-flash" \
    --api_key $API_KEY --base_url $API_BASE
