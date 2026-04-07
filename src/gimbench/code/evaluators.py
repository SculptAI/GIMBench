import contextlib
import math
import os
import subprocess
import sys
import tempfile

from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Literal

from datasets import Dataset
from gimkit.schemas import MaskedTag
from pydantic import BaseModel
from tqdm import tqdm

from gimbench.base import BaseEvalResult, BaseEvaluator
from gimbench.log import get_logger
from gimbench.models import SimpleCommon, SimpleGIM


logger = get_logger(__name__)

_INFILL_MASK = MaskedTag(name="infill", desc="Fill in the missing code")


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k given n total samples and c correct ones (unbiased estimator)."""
    if n < k:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


class EvalItemResult(BaseModel):
    task_id: str = ""
    subset: str = ""
    query: str = ""
    completions: list[str] = []
    full_codes: list[str] = []
    passed: list[bool] = []
    exec_errors: list[str] = []

    error_msg: str = ""


class EvalResult(BaseEvalResult):
    evaluator_type: Literal["code"] = "code"

    total: int
    evaluates: int
    errors: int
    num_samples: int
    pass_at_k: dict[str, float]

    evaled_items: list[EvalItemResult]


class CodeInfillingEvaluator(BaseEvaluator):
    def __init__(self, args: Namespace, dataset: Dataset):
        super().__init__(args, dataset)

    def _execute_code(self, code: str) -> tuple[bool, str]:
        """Write code to a temp file, execute it, and return (passed, error_message)."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as f:
            f.write(code)
            fname = f.name
        try:
            result = subprocess.run(
                [sys.executable, fname],
                capture_output=True,
                text=True,
                timeout=self.args.exec_timeout,
            )
            passed = result.returncode == 0
            error = result.stderr.strip() if not passed else ""
            return passed, error
        except subprocess.TimeoutExpired:
            return False, "TimeoutExpired"
        except Exception as e:
            return False, str(e)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(fname)

    def _build_test_code(self, full_code: str, test: str, entry_point: str) -> str:
        return f"{full_code}\n\n{test}\n\ncheck({entry_point})"

    def _generate_completions(self, item: dict) -> tuple[str, list[str]]:
        """Return (query_string, list_of_infill_completions). Subclasses must implement this."""
        raise NotImplementedError

    def _evaluate_item(self, item: dict) -> EvalItemResult:
        task_id = item.get("task_id", "")
        subset = item.get("subset", "")
        prompt = item["prompt"]
        suffix = item["suffix"]
        test = item["test"]
        entry_point = item["entry_point"]

        try:
            query, completions = self._generate_completions(item)
        except Exception as e:
            logger.error(f"Error generating completions for {task_id}: {e}")
            return EvalItemResult(task_id=task_id, subset=subset, error_msg=str(e))

        full_codes = []
        passed_list = []
        exec_errors = []
        for completion in completions:
            full_code = prompt + completion + suffix
            test_code = self._build_test_code(full_code, test, entry_point)
            passed, exec_error = self._execute_code(test_code)
            full_codes.append(full_code)
            passed_list.append(passed)
            exec_errors.append(exec_error)

        return EvalItemResult(
            task_id=task_id,
            subset=subset,
            query=query,
            completions=completions,
            full_codes=full_codes,
            passed=passed_list,
            exec_errors=exec_errors,
            error_msg="",
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

        self.end_time = datetime.now()
        logger.info(f"Evaluation completed at {self.end_time}")

        errors = sum(1 for item in evaled_items if item.error_msg)
        non_error_items = self._filter_non_error_items(evaled_items)

        pass_at_k_results: dict[str, float] = {}
        for k in self.args.pass_k:
            if k <= self.args.num_samples:
                scores = [_pass_at_k(len(item.passed), sum(item.passed), k) for item in non_error_items]
                pass_at_k_results[f"pass@{k}"] = sum(scores) / len(scores) if scores else 0.0
            else:
                logger.warning(f"Skipping pass@{k}: k={k} > num_samples={self.args.num_samples}")

        for k_name, v in pass_at_k_results.items():
            logger.info(f"{k_name}: {v:.4f}")

        return EvalResult(
            total=total,
            evaluates=len(evaled_items),
            errors=errors,
            num_samples=self.args.num_samples,
            pass_at_k=pass_at_k_results,
            start_time=self.start_time,
            end_time=self.end_time,
            elapsed_minutes=(self.end_time - self.start_time).total_seconds() / 60.0,
            args=self.args,
            evaled_items=evaled_items,
        )


class GIMCodeInfillingEvaluator(CodeInfillingEvaluator):
    """Uses GIMKit constrained decoding to fill in the masked span."""

    def __init__(self, args: Namespace, dataset: Dataset):
        super().__init__(args, dataset)
        self.model = SimpleGIM(args)

    def _generate_completions(self, item: dict) -> tuple[str, list[str]]:
        prompt = item["prompt"]
        suffix = item["suffix"]
        query = f"{prompt}{_INFILL_MASK}{suffix}"
        completions = []
        for _ in range(self.args.num_samples):
            result = self.model.generate(query)
            completions.append(result.tags["infill"].content or "")
        return query, completions


class CommonCodeInfillingEvaluator(CodeInfillingEvaluator):
    """Uses a plain LLM (no GIMKit) to generate the infill via a chat prompt."""

    def __init__(self, args: Namespace, dataset: Dataset):
        super().__init__(args, dataset)
        self.model = SimpleCommon(args)

    def _form_query(self, prompt: str, suffix: str) -> str:
        return (
            "Fill in the missing code between the prefix and suffix.\n"
            "Return ONLY the missing code, wrapped in a markdown code fence:\n"
            "```python\n<missing code>\n```\n"
            "Do not include explanations or any text outside the code fence.\n\n"
            f"Prefix:\n```{prompt}```\n"
            f"Suffix:\n```{suffix}```\n\n"
            "Missing code:"
        )

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove surrounding markdown code fences if present."""
        if text.startswith("```"):
            lines = text.splitlines()
            inner = lines[1:-1] if len(lines) > 1 and lines[-1].startswith("```") else lines[1:]
            return "\n".join(inner)
        return text

    def _generate_completions(self, item: dict) -> tuple[str, list[str]]:
        prompt = item["prompt"]
        suffix = item["suffix"]
        query = self._form_query(prompt, suffix)
        completions = []
        for _ in range(self.args.num_samples):
            response = self.model.generate(query)
            completions.append(self._strip_code_fences(response))
        return query, completions


def conduct_eval(args: Namespace, ds: Dataset) -> None:
    evaluator = CommonCodeInfillingEvaluator(args, ds) if args.no_gimkit else GIMCodeInfillingEvaluator(args, ds)
    result = evaluator.evaluate()
    result.dump()
