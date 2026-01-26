import re

from abc import abstractmethod
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Literal

from datasets import Dataset
from gimkit import from_vllm, guide
from gimkit.contexts import Result
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from gimbench.base import BaseEvalResult, BaseEvaluator
from gimbench.log import get_logger


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

    evaled_items: list[EvalItemResult]


class MCQAEvaluator(BaseEvaluator):
    def __init__(self, args: Namespace, dataset: Dataset):
        super().__init__(args, dataset)

        self._counter_tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(args.counter_tokenizer)
        logger.info(f"Loaded tokenizer {args.counter_tokenizer} for token counting.")

    @abstractmethod
    def _form_cot_query(self, question: str, choices: list[str]) -> str: ...

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
        query = self._form_cot_query(question, choices)
        try:
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
            error_msg=error_msg,
            additional_info=additional_info,
        )

    def evaluate(self) -> EvalResult:
        logger.info(f"Starting evaluation with config: {self.args}")
        total = len(self.dataset) if self.args.first_n == -1 else min(self.args.first_n, len(self.dataset))

        evaled_items = []
        if self.args.num_proc <= 1:
            for idx in tqdm(range(total), desc=f"Evaluating {self.args.model_name}"):
                result = self._evaluate_item(self.dataset[idx])
                evaled_items.append(result)
        else:
            with ThreadPoolExecutor(max_workers=self.args.num_proc) as executor:
                results = executor.map(self._evaluate_item, (self.dataset[i] for i in range(total)))
                evaled_items = list(tqdm(results, total=total, desc=f"Evaluating {self.args.model_name}"))

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
            start_time=self.start_time,
            end_time=self.end_time,
            elapsed_minutes=(self.end_time - self.start_time).total_seconds() / 60.0,
            args=self.args,
            evaled_items=evaled_items,
        )

    def _count_tokens(self, text: str) -> int:
        return len(self._counter_tokenizer.encode(text))


SHARED_PROMPT_PREFIX = (
    "Answer the following question carefully using a variety of strategies, such as reasoning, reflection, "
    "trial and error, and parallel thinking (applying different approaches). "
    "Feel free to use any other methods as needed to find the correct answer. "
    "Verify your work before concluding."
)


class GIMEvaluator(MCQAEvaluator):
    def __init__(self, args: Namespace, dataset: Dataset):
        super().__init__(args, dataset)
        openai_client = OpenAI(api_key=args.api_key, base_url=args.base_url)
        self.model = from_vllm(openai_client, model_name=args.model_name)

    def _get_reason_budget(self, question: str) -> int:
        if self.args.auto_budget:
            try:
                r = self.model.generate(
                    "I'll show you a couple of questions. "
                    "Decide how many reasoning steps are needed to answer each accurately.\n\n"
                    "Consider a plausible reasoning workflow first (you may use reasoning, reflection, "
                    "trial and error, and parallel thinking by applying different approaches, plus a quick verification if needed). "
                    "Then output a step budget (where each step is an atomic reasoning action taking 3–5 sentences) that allows for granular, step-by-step derivation without skipping logic, ensuring a robust and high-confidence conclusion;"
                    "leave extra headroom for cross-checking and possible revision on multi-hop or tricky questions.\n\n"
                    "## Question: A train travels 150 km at 60 km/h, stops for 10 minutes, then continues another 90 km at 45 km/h. "
                    "How long does the full trip take in minutes?\n\n"
                    "## Reasoning steps: 5\n\n"
                    "## Question: An account starts with $5,000. At the beginning of each month for 12 months, a $5 fee is charged. "
                    "At the end of each month, you deposit $200. The annual interest rate is 6% compounded monthly for the first 6 months "
                    "and 4% compounded monthly for the last 6 months. What is the final balance after 12 months, rounded to the nearest cent?\n\n"
                    "## Reasoning steps: 15\n\n"
                    "Do not be anchored by the examples above. Scale your step budget linearly with the difficulty. "
                    "For complex problems, you are encouraged to assign a high budget (20, or more) to ensure there is enough room for step-by-step derivation and verification.\n\n"
                    f"## Question: {question}\n\n"
                    "## Reasoning steps: " + guide(name="reason_budget", desc="A positive integer number", regex=r"\d+")
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
            f"## Step {idx + 1}\n\n" + guide(desc="A distinct, verified reasoning step building on the previous one. Write 2–3 substantial sentences (60–80 words each) to ensure depth.")
            for idx in range(reason_budget)
        ]
        prompt = SHARED_PROMPT_PREFIX + f"\n\nQuestion: {question}\n\n"
        if reason_budget > 0:
            prompt += (
                f"You have {reason_budget} steps maximum. Use each step for a distinct line of reasoning.\n\n"
                "Let's think step by step.\n\n" + "\n\n".join(reasoning_guides) + "\n\n"
            )
        prompt += "## Conclusion\n\nFinal answer: " + guide.select(choices=choices, name="predicted_choice")
        return prompt

    def _model_call(self, query: str) -> Result:
        result = self.model(
            query,
            temperature=self.args.temperature,
            presence_penalty=self.args.presence_penalty,
            seed=self.args.seed,
            max_tokens=self.args.max_tokens,
        )
        if isinstance(result, list):
            raise ValueError("Expected a single Result, but got a list.")
        return result

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

    def _form_cot_query(self, question: str, choices: list[str]) -> str:
        prompt = SHARED_PROMPT_PREFIX + (
            " Remember to end with `The answer is: xxx`.\n\n"
            "Do not write anything after that final line.\n\n"
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

        lines = response_str.splitlines() if response_str else []

        validate_choices_norm = [c.upper() for c in validate_choices]
        options = "".join(validate_choices_norm)

        raw_tail_lines = lines[-5:] if lines else [response_str]

        def _is_code_fence(s: str) -> bool:
            t = s.strip()
            return bool(re.fullmatch(r"`{3,}|~{3,}", t))

        meaningful_tail_lines: list[str] = [ln for ln in raw_tail_lines if ln.strip() and not _is_code_fence(ln)]
        tail_text = "\n".join(meaningful_tail_lines if meaningful_tail_lines else raw_tail_lines).strip()

        def _search_last(pattern: str, text: str, flags: int = 0) -> re.Match | None:
            ms = list(re.finditer(pattern, text, flags))
            return ms[-1] if ms else None

        # 1) Marker-based extraction in tail (take the LAST match)
        marker_pat = (
            rf"(?:the answer is|the correct answer is|final answer|answer)"
            rf"[:\s]*"
            rf"(?:option\s*)?"
            rf"\(?\**\s*([{options}])\s*\**\)?"
        )
        if m := _search_last(marker_pat, tail_text, re.IGNORECASE):
            model_choice = m.group(1).strip().rstrip(".),")
            additional_info["extracted_by"] = "tail_marker_last"
            additional_info["matched_span"] = m.group(0)

        # 1b) LaTeX boxed in tail (take the LAST match)
        elif m_box := _search_last(rf"boxed\{{\s*([{options}])\s*\}}", tail_text, re.IGNORECASE):
            model_choice = m_box.group(1).strip()
            additional_info["extracted_by"] = "tail_marker_boxed_last"
            additional_info["matched_span"] = m_box.group(0)

        # 1c) ANSWER: X in tail (take the LAST match)
        elif m_ans := _search_last(rf"(?i)answer\s*:\s*\(?([{options}])\)?", tail_text):
            model_choice = m_ans.group(1).strip()
            additional_info["extracted_by"] = "tail_marker_answer_colon_last"
            additional_info["matched_span"] = m_ans.group(0)

        # 2) Scan from bottom: allow a single-token line like "A", "(A)", "A.", "A)"
        else:
            token_pat = rf"^\(?\**\s*([{options}])\s*\**\)?[.)]?$"
            picked_line = None
            for ln in reversed(meaningful_tail_lines if meaningful_tail_lines else raw_tail_lines):
                s = ln.strip()
                if not s or _is_code_fence(s):
                    continue
                if m2 := re.match(token_pat, s):
                    model_choice = m2.group(1).strip().rstrip(".),")
                    picked_line = ln
                    additional_info["extracted_by"] = "tail_line_scan_bottom"
                    additional_info["matched_line"] = picked_line
                    break

        if model_choice == "ERROR":
            additional_info["tail_used"] = tail_text
            raise ValueError(f"Could not extract a valid choice from the model response: {response_str}")

        model_choice = model_choice.upper()

        if model_choice not in validate_choices_norm:
            raise ValueError(f"Extracted choice '{model_choice}' not in valid choices {validate_choices}")

        return response_str, model_choice, additional_info


def conduct_eval(args: Namespace, ds: Dataset):
    evaluator = CommonEvaluator(args, ds) if args.no_gimkit else GIMEvaluator(args, ds)
    result = evaluator.evaluate()
    result.dump()
