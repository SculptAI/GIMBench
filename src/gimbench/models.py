import argparse
import time

from typing import Any

from gimkit.contexts import Result


class SimpleGIM:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model: Any
        self._openai_client: Any = None
        if args.model_type in ["openai", "vllm"]:
            from gimkit import from_openai
            from openai import OpenAI as OpenAIClient

            openai_client = OpenAIClient(api_key=args.api_key, base_url=args.base_url)
            self._openai_client = openai_client
            self.model = from_openai(openai_client, args.model_name)
        elif args.model_type == "vllm-offline":
            from gimkit import from_vllm_offline
            from vllm import LLM

            vllm_client = LLM(args.model_name, max_model_len=args.max_model_len)
            self.model = from_vllm_offline(vllm_client)
        else:
            raise ValueError("Unsupported model type")

    def _measure_ttft(self, prompt: str) -> float:
        """Measure time-to-first-token via a single-token streaming API call.

        Returns the elapsed seconds until the first token is received,
        or -1.0 if streaming is unavailable (e.g. vllm-offline) or fails.
        """
        if self._openai_client is None:
            return -1.0
        try:
            start = time.perf_counter()
            with self._openai_client.chat.completions.create(
                model=self.args.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                max_tokens=1,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
            ) as stream:
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        return time.perf_counter() - start
        except Exception:
            pass
        return -1.0

    def generate_with_timing(self, prompt: str) -> tuple[Result, float, float]:
        """Generate a response and collect timing metrics.

        Returns a ``(result, generation_time_seconds, ttft_seconds)`` tuple.
        ``ttft_seconds`` is -1.0 for vllm-offline models or when measurement fails.
        ``generation_time_seconds`` is the wall-clock time for the full generation call.
        """
        ttft = self._measure_ttft(prompt)
        start = time.perf_counter()
        result = self.generate(prompt)
        generation_time = time.perf_counter() - start
        return result, generation_time, ttft

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
