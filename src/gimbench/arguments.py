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
    parser.add_argument("--model_name", type=str, required=True, help="Model under evaluation")
    parser.add_argument("--api_key", type=str, default="", help="API key for the model")
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for the model API",
    )


def _add_sample_args(parser):
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for the model")
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
    parser.add_argument(
        "--counter_tokenizer",
        type=str,
        default="unsloth/Qwen3-4B-Instruct-2507",
        help="Tokenizer to use for token counting",
    )


def _add_ctp_eval_args(parser):
    parser.add_argument(
        "--ref_model_name",
        type=str,
        default="gpt2",
        help="Reference model for Composite Text Perplexity (CTP) evaluation",
    )
    parser.add_argument(
        "--ref_model_device",
        type=str,
        default="cpu",
        help="Device for the reference model",
    )
    parser.add_argument(
        "--base_model_vocab_size",
        type=int,
        default=0,
        help="Vocabulary size of the base model for Normalized CTP calculation",
    )
    parser.add_argument(
        "--ctp_alpha",
        type=float,
        default=0.2,
        help="Scaling factor alpha for Normalized CTP",
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
    _add_ctp_eval_args(parser)
    _add_mcqa_eval_args(parser)
    args = parser.parse_args()
    validate_and_standardize(args)
    return args
