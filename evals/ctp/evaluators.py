import os
import subprocess

from abc import abstractmethod
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from datasets import Dataset
from gimkit import from_vllm
from gimkit.contexts import Query, Result
from openai import OpenAI
from pydantic import BaseModel, field_serializer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from evals.log import get_logger


GIT_BRANCH = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode("utf-8")
GIT_COMMIT_ID = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")

logger = get_logger(__name__)


class EvalItemResult(BaseModel):
    query: str = ""
    result: str = ""

    ctp: float = -1.0
    query_tags: int = -1
    result_tags: int = -1
    infilling_ratio: float = -1.0

    error_msg: str = ""


class EvalResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    evaluator_type: str = "ctp"

    total: int
    evaluates: int
    errors: int

    avg_ctp: float = 0.0
    avg_query_tags: float = 0.0
    avg_result_tags: float = 0.0
    avg_infilling_ratio: float = 0.0

    start_time: datetime
    end_time: datetime
    elapsed_minutes: float = 0.0
    args: Namespace
    git_branch: str = GIT_BRANCH
    git_commit_id: str = GIT_COMMIT_ID
    evaled_items: list[EvalItemResult] = []

    @field_serializer("args")
    def serialize_args(self, value: Namespace) -> dict[str, Any]:
        return vars(value)

    def dump(self, filepath: str | None = None):
        if filepath is None:
            dataset = getattr(self.args, "dataset", {})
            dataset_path = dataset.get("path", "unknown_dataset") if isinstance(dataset, dict) else "unknown_dataset"
            model_name = getattr(self.args, "model_name", "unknown_model")
            filename = f"{model_name}_{dataset_path}_{self.start_time.strftime('%y%m%d-%H%M%S')}.json".replace("/", "_")
            filepath = Path(self.args.output_dir) / filename
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(self.model_dump_json(indent=4))
        logger.info(f"Saved evaluation results to {filepath}")


class BaseEvaluator:
    def __init__(self, args: Namespace, dataset: Dataset):
        assert "gim_query" in dataset.column_names, "Dataset must contain 'gim_query' column"

        self.start_time = datetime.now()
        self.dataset = dataset
        self.args = args
        self.ref_model = AutoModelForCausalLM.from_pretrained(args.ref_model_name).to(self.args.ref_model_device)
        self.ref_tokenizer = AutoTokenizer.from_pretrained(args.ref_model_name)

    @abstractmethod
    def _model_call(self, query: str) -> str:
        """Call the model with the given query and return the response."""

    @abstractmethod
    def _compute_ctp(self, text: str) -> float:
        """Compute Composite Text Perplexity (CTP) for the given text."""

    def _evaluate_item(self, item: dict) -> EvalItemResult:
        result = "ERROR"
        ctp = -1.0
        error_msg = ""
        try:
            query = str(Query(item["gim_query"]))
            result = self._model_call(query)
            ctp = self._compute_ctp(result)
        except IndexError:
            err_msg = f"{self.args.model_name}'s context window may be too small for CTP evaluation."
            logger.error(err_msg)
            error_msg = err_msg
        except Exception as e:
            logger.exception(e)
            error_msg = str(e)
        return EvalItemResult(
            query=query,
            result=result,
            ctp=ctp,
            query_tags=len(Query(query).tags),
            result_tags=len(Result(result).tags),
            infilling_ratio=(1 - len(Result(result).tags) / len(Query(query).tags))
            if len(Query(query).tags) > 0
            else -1.0,
            error_msg=error_msg,
        )

    def evaluate(self) -> EvalResult:
        logger.info(f"Starting evaluation with config: {self.args}")
        total = len(self.dataset) if self.args.first_n == -1 else min(self.args.first_n, len(self.dataset))

        evaled_items = []
        for idx in tqdm(range(total), desc=f"Evaluating {self.args.model_name}"):
            result = self._evaluate_item(self.dataset[idx])
            evaled_items.append(result)

        self.end_time = datetime.now()
        logger.info(f"Evaluation completed at {self.end_time}")

        def safe_average(items: list[EvalItemResult], attr: str) -> float:
            values = [getattr(item, attr) for item in items if getattr(item, attr) != -1]
            return sum(values) / len(values) if values else 0.0

        return EvalResult(
            total=total,
            evaluates=len(evaled_items),
            errors=sum(1 for item in evaled_items if item.error_msg),
            avg_ctp=safe_average(evaled_items, "ctp"),
            avg_query_tags=safe_average(evaled_items, "query_tags"),
            avg_result_tags=safe_average(evaled_items, "result_tags"),
            avg_infilling_ratio=safe_average(evaled_items, "infilling_ratio"),
            start_time=self.start_time,
            end_time=self.end_time,
            elapsed_minutes=(self.end_time - self.start_time).total_seconds() / 60.0,
            args=self.args,
            evaled_items=evaled_items,
        )


class GIMEvaluator(BaseEvaluator):
    def __init__(self, args: Namespace, dataset: Dataset):
        super().__init__(args, dataset)
        openai_client = OpenAI(api_key=args.api_key, base_url=args.base_url)
        self.model = from_vllm(openai_client, model_name=args.model_name)

    def _model_call(self, query: str) -> str:
        result = self.model(
            query,
            temperature=self.args.temperature,
            presence_penalty=self.args.presence_penalty,
            seed=self.args.seed,
            max_tokens=self.args.max_tokens,
        )
        return str(result)

    def _compute_ctp(self, text: str) -> float:
        tokens = self.ref_tokenizer(text, return_tensors="pt").input_ids.to(self.args.ref_model_device)
        with torch.no_grad():
            outputs = self.ref_model(tokens, labels=tokens)
            loss = outputs.loss
        perplexity = torch.exp(loss).item()
        return perplexity


def conduct_eval(args: Namespace, ds: Dataset):
    if not args.is_gim:
        raise NotImplementedError("Only GIM evaluation is implemented in this evaluator.")
    evaluator = GIMEvaluator(args, ds)
    result = evaluator.evaluate()
    result.dump()
