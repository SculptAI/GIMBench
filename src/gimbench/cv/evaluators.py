from abc import abstractmethod
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Literal

import json_repair

from datasets import Dataset
from gimkit import guide as g
from gimkit.contexts import Result
from outlines.types import JsonSchema
from pydantic import BaseModel, Field
from tqdm import tqdm

from gimbench.base import BaseEvalResult, BaseEvaluator
from gimbench.log import get_logger
from gimbench.models import SimpleGIM

from .schema import CV_FIELDS, GIMKIT_TEMPLATE, OUTLINES_JSON_SCHEMA, OUTLINES_TEMPLATE, SHARED_PROMPT_PREFIX


logger = get_logger(__name__)


class EvalItemResult(BaseModel):
    filename: str = ""
    extraction_details: dict[str, dict] = Field(default_factory=dict)
    num_fields: int = -1
    num_correct: int = -1

    error_msg: str = ""


class EvalResult(BaseEvalResult):
    evaluator_type: Literal["cv"] = "cv"

    total_fields: int
    total_correct: int
    errors: int
    calibrated_accuracy: float

    evaled_items: list[EvalItemResult]


class CVEvaluator(BaseEvaluator):
    def __init__(self, args: Namespace, dataset: Dataset):
        super().__init__(args, dataset)

        self.judge_model = SimpleGIM(
            Namespace(
                model_type="openai",
                model_name=args.judge_model_name,
                api_key=args.api_key,
                base_url=args.base_url,
                output_type="json",
                use_gim_prompt=True,
                temperature=0.0,
                presence_penalty=1.0,
                max_tokens=256,
            )
        )

    @abstractmethod
    def _extract_fields(self, cv_content: str) -> dict[str, str]:
        """Extract the CV fields from the given content."""
        ...

    def _judge(self, extracted_fields: dict[str, str], eval_details: dict[str, dict]) -> None:
        """Judge the extracted fields against expected values and update details in place."""

        def _judge_field_with_model(field: str, predicted: str, expected: str) -> bool:
            result = self.judge_model.generate(
                f"You are evaluating the accuracy of extracted information from a CV. "
                f"Please determine if the extracted value for the field '{field}' is semantically equivalent to the expected value. "
                f"\n\nField: {field}\nExtracted Value: '{predicted}'\nExpected Value: '{expected}'\n"
                f"\nIs the extracted value correct?\n"
                f"Answer: {g.select(choices=['Correct', 'Incorrect'])}"
            )
            logger.info(
                f"Judgment for field '{field}', predicted: '{predicted}', expected: '{expected}': {result.tags[0].content}"
            )
            return result.tags[0].content == "Correct"

        for field, info in eval_details.items():
            predicted = extracted_fields.get(field, "")
            expected = info["expected"]
            info["prediction"] = predicted
            if predicted == "" or expected == "":
                info["verbatim_correct"] = False
            elif predicted == expected:
                info["verbatim_correct"] = True
            else:
                info["judge_model_correct"] = _judge_field_with_model(field, predicted, expected)
            info["correct"] = info["verbatim_correct"] or info["judge_model_correct"]

    def _evaluate_item(self, item: dict) -> EvalItemResult:
        cv_content = item["extracted_text"]
        eval_details = {
            field: {
                "prediction": "N/A",
                "expected": item.get(field, ""),
                "verbatim_correct": False,
                "judge_model_correct": False,
                "correct": False,
            }
            for field in CV_FIELDS
        }
        try:
            extracted_fields = self._extract_fields(cv_content)
            self._judge(extracted_fields, eval_details)
            error_msg = ""
        except Exception as e:
            logger.error(e)
            error_msg = str(e)
        return EvalItemResult(
            filename=item["file_name"],
            extraction_details=eval_details,
            num_fields=sum(1 for v in eval_details.values() if v["expected"]),
            num_correct=sum(1 for v in eval_details.values() if v["correct"]),
            error_msg=error_msg,
        )

    def evaluate(self) -> EvalResult:
        logger.info(f"Starting evaluation with config: {self.args}")
        total = len(self.dataset) if self.args.first_n == -1 else min(self.args.first_n, len(self.dataset))

        evaled_items = []
        if self.args.num_proc <= 1 or self.args.model_type not in ["openai", "vllm"]:
            for idx in tqdm(range(total), desc=f"Evaluating {self.args.model_name}"):
                result = self._evaluate_item(self.dataset[idx])
                evaled_items.append(result)

                self._log_progress(total, idx)
        else:
            with ThreadPoolExecutor(max_workers=self.args.num_proc) as executor:
                results = executor.map(self._evaluate_item, (self.dataset[i] for i in range(total)))
                evaled_items = list(tqdm(results, total=total, desc=f"Evaluating {self.args.model_name}"))
            # TODO: Add progress logging for multi-threaded evaluation

        non_error_items = [item for item in evaled_items if not item.error_msg]
        errors = sum(1 for item in evaled_items if item.error_msg)
        total_fields = sum(item.num_fields for item in non_error_items)
        total_correct = sum(item.num_correct for item in non_error_items)
        calibrated_accuracy = total_correct / total_fields if total_fields > 0 else 0.0

        logger.info(f"Final accuracy over {total} cvs: {total_correct}/{total_fields} = {calibrated_accuracy:.4f}")
        self.end_time = datetime.now()
        logger.info(f"Evaluation completed at {self.end_time}")

        return EvalResult(
            total_fields=total_fields,
            total_correct=total_correct,
            errors=errors,
            calibrated_accuracy=calibrated_accuracy,
            start_time=self.start_time,
            end_time=self.end_time,
            elapsed_minutes=(self.end_time - self.start_time).total_seconds() / 60.0,
            args=self.args,
            evaled_items=evaled_items,
        )


class GIMEvaluator(CVEvaluator):
    def __init__(self, args: Namespace, dataset: Dataset):
        super().__init__(args, dataset)
        self.model = SimpleGIM(args)

    def _model_call(self, query: str) -> Result:
        return self.model.generate(query)

    def _extract_fields(self, cv_content: str) -> dict[str, str]:
        query = SHARED_PROMPT_PREFIX.format(cv_content=cv_content) + GIMKIT_TEMPLATE
        result = self._model_call(query)
        extraction = {}
        for field in CV_FIELDS:
            extracted_text = result.tags[field].content
            extraction[field] = extracted_text if extracted_text else ""
        return extraction


class OutlinesEvaluator(CVEvaluator):
    def __init__(self, args: Namespace, dataset: Dataset):
        super().__init__(args, dataset)
        self.model = self._get_outlines_model(args)

    def _get_outlines_model(self, args: Namespace):
        if args.model_type == "vllm-offline":
            from outlines import from_vllm_offline
            from vllm import LLM

            vllm_client = LLM(args.model_name, max_model_len=args.max_model_len)
            return from_vllm_offline(vllm_client)
        elif args.model_type in ["openai", "vllm"]:
            from openai import OpenAI as OpenAIClient
            from outlines import from_openai

            openai_client = OpenAIClient(api_key=args.api_key, base_url=args.base_url)
            return from_openai(openai_client, args.model_name)
        else:
            raise ValueError("Unsupported model type for OutlinesEvaluator")

    def _model_call(self, query: str, json_schema: dict) -> str:
        if self.args.model_type == "vllm-offline":
            from vllm import SamplingParams

            response = self.model(
                query,
                output_type=JsonSchema(json_schema),
                sampling_params=SamplingParams(
                    temperature=self.args.temperature,
                    presence_penalty=self.args.presence_penalty,
                    max_tokens=self.args.max_tokens,
                ),
            )
        elif self.args.model_type in ["openai", "vllm"]:
            response = self.model(
                query,
                output_type=JsonSchema(json_schema),
                temperature=self.args.temperature,
                presence_penalty=self.args.presence_penalty,
                max_tokens=self.args.max_tokens,
            )
        else:
            raise ValueError("Unsupported model type for OutlinesEvaluator")
        return response

    def _extract_fields(self, cv_content: str) -> dict[str, str]:
        query = SHARED_PROMPT_PREFIX.format(cv_content=cv_content) + OUTLINES_TEMPLATE
        response_content = self._model_call(query, OUTLINES_JSON_SCHEMA)
        extraction = json_repair.loads(response_content)
        if isinstance(extraction, dict):
            return {str(key): ("" if value is None else str(value)) for key, value in extraction.items()}
        else:
            raise ValueError(f"Expected dict but got {type(extraction).__name__}: {extraction}")


def conduct_eval(args: Namespace, ds: Dataset):
    evaluator = OutlinesEvaluator(args, ds) if args.use_outlines else GIMEvaluator(args, ds)
    result = evaluator.evaluate()
    result.dump()
