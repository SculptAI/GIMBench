import re

from abc import abstractmethod
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Literal

from datasets import Dataset
from gimkit import guide
from gimkit.contexts import Result
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from gimbench.base import BaseEvalResult, BaseEvaluator
from gimbench.log import get_logger
from gimbench.models import SimpleGIM


logger = get_logger(__name__)


class EvalItemResult(BaseModel):
    conclusion: bool
    query: str = ""
    response: str = ""
    model_choice: str = ""
    correct_choice: str = ""

    query_tokens: int = -1
    response_tokens: int = -1
    query_len: int = -1
    response_len: int = -1
    reason_budget: int = -1

    error_msg: str = ""
    additional_info: dict = {}


class EvalResult(BaseEvalResult):
    evaluator_type: Literal["mcqa"] = "mcqa"

    total: int
    evaluates: int
    corrects: int
    errors: int

    accuracy: float
    calibrated_accuracy: float
    avg_query_tokens: float
    avg_response_tokens: float
    avg_query_len: float
    avg_response_len: float
    avg_reason_budget: float

    evaled_items: list[EvalItemResult]


class MCQAEvaluator(BaseEvaluator):
    def __init__(self, args: Namespace, dataset: Dataset):
        super().__init__(args, dataset)

        self._counter_tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(args.counter_tokenizer)
        logger.info(f"Loaded tokenizer {args.counter_tokenizer} for token counting.")

    @abstractmethod
    def _get_reason_budget(self, question: str) -> int: ...

    @abstractmethod
    def _form_cot_query(self, question: str, choices: list[str], *args) -> str: ...

    @abstractmethod
    def _model_call(self, query: str) -> Any: ...

    @abstractmethod
    def _parse_response(self, response: Any, validate_choices: list[str]) -> tuple[str, str, dict]:
        """Extract the response string, model choice, and any additional info from the model response."""
        ...

    def _evaluate_item(self, item: dict) -> EvalItemResult:
        question, choices, correct_choice = (
            item["question"],
            item["choices"],
            item["correct_choice"],
        )
        try:
            if self.args.no_gimkit:
                reason_budget = -1
                query = self._form_cot_query(question, choices)
                raw_response = self._model_call(query)
            else:
                reason_budget = self._get_reason_budget(question)
                query = self._form_cot_query(question, choices, reason_budget)
                raw_response = self._model_call(query)
            response, model_choice, additional_info = self._parse_response(raw_response, choices)
            conclusion = model_choice == correct_choice
            error_msg = ""
        except Exception as e:
            logger.error(e)
            conclusion = False
            response = "ERROR"
            model_choice = "ERROR"
            error_msg = str(e)
            additional_info = {}
        return EvalItemResult(
            conclusion=conclusion,
            query=query,
            response=response,
            model_choice=model_choice,
            correct_choice=correct_choice,
            query_tokens=self._count_tokens(query),
            response_tokens=self._count_tokens(response) if response != "ERROR" else -1,
            query_len=len(query),
            response_len=len(response),
            reason_budget=reason_budget,
            error_msg=error_msg,
            additional_info=additional_info,
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

        errors = sum(1 for item in evaled_items if item.error_msg)
        corrects = sum(1 for item in evaled_items if item.conclusion)
        evaluates = len(evaled_items)
        accuracy = corrects / evaluates if evaluates > 0 else 0.0
        calibrated_accuracy = corrects / (evaluates - errors) if (evaluates - errors) > 0 else 0.0
        logger.info(f"Final accuracy over {total} examples: {corrects}/{total} = {accuracy:.4f}")
        self.end_time = datetime.now()
        logger.info(f"Evaluation completed at {self.end_time}")

        return EvalResult(
            total=total,
            evaluates=evaluates,
            corrects=corrects,
            errors=errors,
            accuracy=accuracy,
            calibrated_accuracy=calibrated_accuracy,
            avg_query_tokens=self._safe_average(evaled_items, "query_tokens"),
            avg_response_tokens=self._safe_average(evaled_items, "response_tokens"),
            avg_query_len=self._safe_average(evaled_items, "query_len"),
            avg_response_len=self._safe_average(evaled_items, "response_len"),
            avg_reason_budget=self._safe_average(evaled_items, "reason_budget"),
            start_time=self.start_time,
            end_time=self.end_time,
            elapsed_minutes=(self.end_time - self.start_time).total_seconds() / 60.0,
            args=self.args,
            evaled_items=evaled_items,
        )

    def _count_tokens(self, text: str) -> int:
        return len(self._counter_tokenizer.encode(text))


SHARED_PROMPT_PREFIX = (
    "Answer the following question using a variety of strategies, such as reasoning, reflection, "
    "trial and error, and parallel thinking (applying different approaches). "
    "Feel free to use any other methods as needed to find the correct answer."
)


class GIMEvaluator(MCQAEvaluator):
    def __init__(self, args: Namespace, dataset: Dataset):
        super().__init__(args, dataset)
        self.model = SimpleGIM(args)

    def _get_reason_budget(self, question: str) -> int:
        if self.args.auto_budget:
            try:
                r = self.model.generate(
                    f"I'll show you a question. "
                    f"You need to determine how many reasoning steps are required to accurately answer it.\n\n"
                    f"## Question: Find the sum of first 5 positive integers.\n\n"
                    f"## Reasoning steps: 2\n\n"
                    f"## Question: {question}\n\n"
                    f"## Reasoning steps: "
                    + guide(name="reason_budget", desc="A positive integer number", regex=r"\d+")
                )
                budget = int(r.tags["reason_budget"].content or "1")
            except Exception as e:
                logger.warning(f"Auto-budget determination failed: {e}")
                budget = 1
            reason_budget = max(1, budget)
            logger.info(f"Auto-determined reasoning budget: {reason_budget}")
        else:
            reason_budget = self.args.reason_budget
        return reason_budget

    def _form_cot_query(self, question: str, choices: list[str], reason_budget: int) -> str:
        reasoning_guides = [
            f"## Step {idx + 1}\n\n" + guide(desc="One thinking step. About 60 words") for idx in range(reason_budget)
        ]
        prompt = SHARED_PROMPT_PREFIX + f"\n\nQuestion: {question}\n\n"
        if reason_budget > 0:
            prompt += "Let's think step by step.\n\n" + "\n\n".join(reasoning_guides) + "\n\n"
        prompt += "## Conclusion\n\nFinal answer: " + guide.select(choices=choices, name="predicted_choice")
        return prompt

    def _model_call(self, query: str) -> Result:
        return self.model.generate(query)

    def _parse_response(self, response: Result, validate_choices: list[str]) -> tuple[str, str, dict]:
        str_response = str(response)
        content = response.tags["predicted_choice"].content or ""
        model_choice = content.strip().strip("().,")
        additional_info = {tag.name or str(tag.id): tag.content for tag in response.tags}
        if model_choice not in validate_choices:
            raise ValueError(f"Extracted choice '{model_choice}' not in valid choices {validate_choices}")
        return str_response, model_choice, additional_info


class CommonEvaluator(MCQAEvaluator):
    def __init__(self, args: Namespace, dataset: Dataset):
        super().__init__(args, dataset)
        self.model = OpenAI(api_key=args.api_key, base_url=args.base_url)

    def _get_reason_budget(self, question: str) -> int:
        raise NotImplementedError("CommonEvaluator does not support reason budget.")

    def _form_cot_query(self, question: str, choices: list[str], *args) -> str:
        prompt = SHARED_PROMPT_PREFIX + (
            " Remember to end with `The answer is: xxx`.\n\n"
            f"Question: {question}\n\n"
            f"Choose from the following options: {', '.join(choices)}\n\n"
            "Let's think step by step:\n"
        )
        return prompt

    def _model_call(self, query: str) -> str:
        response = self.model.chat.completions.create(
            model=self.args.model_name, messages=[{"role": "user", "content": query}]
        )
        return response.choices[0].message.content or ""

    def _parse_response(self, response: str, validate_choices: list[str]) -> tuple[str, str, dict]:
        response_str = response.strip()
        model_choice = "ERROR"
        additional_info = {f"line_{i + 1}": line for i, line in enumerate(response_str.splitlines())}

        last_line = response_str.splitlines()[-1] if response_str.splitlines() else response_str

        # 1) Try marker-based extraction: e.g. "The answer is: A", "Final answer: (B)", "Answer: C."
        if m := re.search(
            r"(?:the answer is|final answer|answer)[:\s]*\(?([A-Za-z0-9]+)\)?",
            last_line,
            re.IGNORECASE,
        ):
            model_choice = m.group(1).strip().rstrip(".),")
            additional_info["extracted_by"] = "marker"

        # 2) Scan last line for a short token like "A", "(A)", "A.", "A)" at line start or alone
        elif m2 := re.match(r"^\(?([A-Za-z0-9])\)?[\.|\)]?$", last_line):
            model_choice = m2.group(1).strip().rstrip(".),")
            additional_info["extracted_by"] = "line_scan_last"

        if model_choice == "ERROR":
            raise ValueError(f"Could not extract a valid choice from the model response: {response_str}")

        if model_choice not in validate_choices:
            raise ValueError(f"Extracted choice '{model_choice}' not in valid choices {validate_choices}")

        return response_str, model_choice, additional_info


def conduct_eval(args: Namespace, ds: Dataset):
    evaluator = CommonEvaluator(args, ds) if args.no_gimkit else GIMEvaluator(args, ds)
    result = evaluator.evaluate()
    result.dump()
