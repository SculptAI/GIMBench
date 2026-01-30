import argparse

from argparse import ArgumentParser


SECRET_ARGS = ["api_key"]


def _add_gim_args(parser):
    parser.add_argument("--use_gim_prompt", action="store_true", help="Whether to use GIM prompt")
    parser.add_argument(
        "--output_type",
        type=str,
        choices=["none", "json", "cfg"],
        default="none",
        help="Constrained decoding output type",
    )


def _add_model_args(parser):
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["openai", "vllm", "vllm-offline"],
        help="Type of model to use",
    )
    parser.add_argument("--model_name", type=str, help="Model under evaluation")
    parser.add_argument("--api_key", type=str, default="", help="API key for the model")
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for the model API",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="Maximum length of the vllm model context window",
    )


def _add_sample_args(parser):
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for the model")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling for the model")
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=1.0,
        help="Presence penalty for the model",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Maximum tokens for the model response",
    )


def _add_evaluator_args(parser):
    parser.add_argument("--seed", type=int, default=16, help="Random seed for reproducibility")
    parser.add_argument(
        "--first_n",
        type=int,
        default=-1,
        help="Evaluate only the first n samples. -1 means all",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of processes for parallel evaluation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save evaluation results",
    )


def _add_ppl_eval_args(parser):
    parser.add_argument(
        "--ref_model_name",
        type=str,
        default="google/gemma-3-270m",
        help="Reference model for Perplexity (PPL) evaluation",
    )
    parser.add_argument(
        "--ref_model_device",
        type=str,
        default="cpu",
        help="Device for the reference model",
    )
    parser.add_argument(
        "--norm_ppl_alpha",
        type=float,
        default=0.2,
        help="Scaling factor alpha for Normalized PPL",
    )
    parser.add_argument(
        "--ppl_window_k",
        type=int,
        default=16,
        help="Window size for PPL calculation. The window will be of size 2k",
    )
    parser.add_argument(
        "--golden_truth_only",
        action="store_true",
        help="Only evaluate PPL on golden truth responses",
    )


def _add_mcqa_eval_args(parser):
    parser.add_argument("--no_gimkit", action="store_true", help="Whether to disable GIM kit usage")
    parser.add_argument(
        "--reason_budget",
        type=int,
        default=0,
        help="Number of reasoning steps to include in the prompt",
    )
    parser.add_argument(
        "--auto_budget",
        action="store_true",
        help="Automatically determine the reasoning budget (overrides --reason_budget if both are set)",
    )
    parser.add_argument(
        "--auto_budget_prompt",
        type=str,
        default=(
            "I'll show you a couple of questions. "
            "Decide how many reasoning steps are needed to answer each accurately.\n\n"
            "Consider a plausible reasoning workflow first (you may use reasoning, reflection, "
            "trial and error, and parallel thinking by applying different approaches, plus a quick verification if needed). "
            "Then output a step budget (where each step is an atomic reasoning action taking 3–5 sentences) that allows for granular, step-by-step derivation without skipping logic, ensuring a robust and high-confidence conclusion;"
            "leave extra headroom for cross-checking and possible revision on multi-hop or tricky questions.\n\n"
            "## Question: {question}\n\n"
            "Do not be anchored by the examples above. Scale your step budget linearly with the difficulty. "
            "For complex problems, you are encouraged to assign a high budget (20, or more) to ensure there is enough room for step-by-step derivation and verification.\n\n"
        ),
        help="Template prompt for auto budget determination; use {question} as placeholder for the question.",
    )
    parser.add_argument(
        "--reason_step_desc",
        type=str,
        default=(
            "A distinct, verified reasoning step building on the previous one. "
            "Write 2–3 substantial sentences (60–80 words each) to ensure depth."
        ),
        help="Description template for each reasoning step used by guide(desc=...).",
    )
    parser.add_argument(
        "--counter_tokenizer",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Tokenizer to use for token counting",
    )


def _add_cv_eval_args(parser):
    parser.add_argument("--use_outlines", action="store_true", help="Whether to use outlines in CV evaluation")
    parser.add_argument(
        "--judge_model_name",
        type=str,
        default="google/gemini-3-flash-preview",
        help="Model name for judgment in CV evaluation. Only API-based models are supported.",
    )


def validate_and_standardize(args: argparse.Namespace) -> argparse.Namespace:
    if args.model_type == "openai" and not (args.api_key and args.base_url):
        raise ValueError("API key and base URL must be provided for OpenAI models.")
    if args.model_type == "vllm" and not args.base_url:
        raise ValueError("Base URL must be provided for vLLM models.")

    if args.output_type == "none":
        args.output_type = None
    return args


def get_args() -> argparse.Namespace:
    parser = ArgumentParser()
    _add_gim_args(parser)
    _add_model_args(parser)
    _add_sample_args(parser)
    _add_evaluator_args(parser)
    _add_ppl_eval_args(parser)
    _add_mcqa_eval_args(parser)
    _add_cv_eval_args(parser)
    args = parser.parse_args()
    validate_and_standardize(args)
    return args
