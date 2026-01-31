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

            vllm_client = LLM(args.model_name, max_model_len=args.max_model_len)
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
                top_p=self.args.top_p,
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
                    top_p=self.args.top_p,
                    max_tokens=self.args.max_tokens,
                    presence_penalty=self.args.presence_penalty,
                ),
            )
        else:
            raise ValueError("Unsupported model type")


class SimpleCommon:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model: Any
        if args.model_type in ["openai", "vllm"]:
            from openai import OpenAI as OpenAIClient

            self.model = OpenAIClient(api_key=args.api_key, base_url=args.base_url)
        elif args.model_type == "vllm-offline":
            from vllm import LLM

            self.model = LLM(args.model_name, max_model_len=args.max_model_len)
        else:
            raise ValueError("Unsupported model type")

    def generate(self, prompt: str) -> str:
        if self.args.model_type in ["openai", "vllm"]:
            response = self.model.chat.completions.create(
                model=self.args.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                presence_penalty=self.args.presence_penalty,
                max_tokens=self.args.max_tokens,
            )
            return response.choices[0].message.content or ""
        elif self.args.model_type == "vllm-offline":
            from vllm import SamplingParams

            outputs = self.model.generate(
                prompt,
                sampling_params=SamplingParams(
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    max_tokens=self.args.max_tokens,
                    presence_penalty=self.args.presence_penalty,
                ),
            )
            for output in outputs:
                prompt = output.prompt
                response = output.outputs[0].text
            return response or ""

        else:
            raise ValueError("Unsupported model type")
