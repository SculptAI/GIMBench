import argparse

from typing import Any

from gimkit.contexts import Result


class SimpleGIM:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model: Any
        if args.model_type in ["openai", "vllm"]:
            from gimkit import from_openai
            from openai import OpenAI as OpenAIClient

            openai_client = OpenAIClient(api_key=args.api_key, base_url=args.base_url)
            self.model = from_openai(openai_client, args.model_name)
        elif args.model_type == "vllm-offline":
            from gimkit import from_vllm_offline
            from vllm import LLM

            vllm_client = LLM(args.model_name)
            self.model = from_vllm_offline(vllm_client)
        else:
            raise ValueError("Unsupported model type")

    def generate(self, prompt: str) -> Result:
        if self.args.model_type in ["openai", "vllm"]:
            return self.model(
                prompt,
                output_type=self.args.output_type,
                use_gim_prompt=self.args.use_gim_prompt,
                temperature=self.args.temperature,
                presence_penalty=self.args.presence_penalty,
                max_tokens=self.args.max_tokens,
            )
        elif self.args.model_type == "vllm-offline":
            from vllm import SamplingParams

            return self.model(
                prompt,
                output_type=self.args.output_type,
                use_gim_prompt=self.args.use_gim_prompt,
                sampling_params=SamplingParams(
                    temperature=self.args.temperature,
                    max_tokens=self.args.max_tokens,
                    presence_penalty=self.args.presence_penalty,
                ),
            )
        else:
            raise ValueError("Unsupported model type")
