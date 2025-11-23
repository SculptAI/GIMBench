from abc import abstractmethod
from argparse import Namespace
from datetime import datetime
from typing import Literal

import torch

from datasets import Dataset
from gimkit import from_vllm
from gimkit.contexts import Query, Result
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from gimbench.base import BaseEvalResult, BaseEvaluator
from gimbench.log import get_logger


logger = get_logger(__name__)


class EvalItemResult(BaseModel):
    query: str = ""
    result: str = ""

    ctp: float = -1.0
    query_tags: int = -1
    result_tags: int = -1
    infilling_ratio: float = -1.0

    error_msg: str = ""


class EvalResult(BaseEvalResult):
    evaluator_type: Literal["ctp"] = "ctp"

    total: int
    evaluates: int
    errors: int

    avg_ctp: float = 0.0
    avg_query_tags: float = 0.0
    avg_result_tags: float = 0.0
    avg_infilling_ratio: float = 0.0

    evaled_items: list[EvalItemResult]


class CTPEvaluator(BaseEvaluator):
    def __init__(self, args: Namespace, dataset: Dataset):
        if "gim_query" not in dataset.column_names:
            raise ValueError("Dataset must contain 'gim_query' column for CTP evaluation.")

        super().__init__(args, dataset)
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
            error_msg = f"{self.args.model_name}'s context window may be too small for CTP evaluation."
            logger.error(error_msg)
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

        return EvalResult(
            total=total,
            evaluates=len(evaled_items),
            errors=sum(1 for item in evaled_items if item.error_msg),
            avg_ctp=self._safe_average(evaled_items, "ctp"),
            avg_query_tags=self._safe_average(evaled_items, "query_tags"),
            avg_result_tags=self._safe_average(evaled_items, "result_tags"),
            avg_infilling_ratio=self._safe_average(evaled_items, "infilling_ratio"),
            start_time=self.start_time,
            end_time=self.end_time,
            elapsed_minutes=(self.end_time - self.start_time).total_seconds() / 60.0,
            args=self.args,
            evaled_items=evaled_items,
        )


class GIMEvaluator(CTPEvaluator):
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
    if args.no_gimkit:
        raise NotImplementedError("Only GIM evaluation is implemented in this evaluator.")
    evaluator = GIMEvaluator(args, ds)
    result = evaluator.evaluate()
    result.dump()
