import logging
import re

from argparse import Namespace
from datetime import datetime
from typing import Literal

from datasets import Dataset
from gimkit.contexts import Query
from gimkit.schemas import MaskedTag
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from gimbench.base import BaseEvalResult, BaseEvaluator
from gimbench.log import get_logger
from gimbench.models import SimpleGIM


logging.getLogger("gimkit").setLevel(logging.DEBUG)

logger = get_logger(__name__)


class EvalItemResult(BaseModel):
    query: str
    result: str
    tags: list[MaskedTag]
    regex_matched_ids: list[int]
    num_tags: int
    num_has_prediction: int
    num_regex: int
    num_regex_match: int

    response_tokens: int = -1
    generation_time: float = -1.0
    throughput: float = -1.0
    ttft: float = -1.0

    error_msg: str = ""


class EvalResult(BaseEvalResult):
    evaluator_type: Literal["match"] = "match"

    total_queries: int
    errors: int

    total_tags: int
    valid_tags: int
    total_has_prediction: int
    total_regex: int
    valid_regex: int
    total_regex_match: int

    prediction_rate: float
    match_rate: float

    avg_response_tokens: float = 0.0
    avg_generation_time: float = 0.0
    avg_throughput: float = 0.0
    avg_ttft: float = 0.0

    evaled_items: list[EvalItemResult]


class MatchEvaluator(BaseEvaluator):
    def __init__(self, args: Namespace, dataset: Dataset):
        if args.no_gimkit:
            raise ValueError("GIMKit must be enabled for MatchEvaluator.")

        super().__init__(args, dataset)
        self.model = SimpleGIM(args)
        self._counter_tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(args.counter_tokenizer)
        logger.info(f"Loaded tokenizer {args.counter_tokenizer} for token counting.")

    def _count_tokens(self, text: str) -> int:
        return len(self._counter_tokenizer.encode(text))

    def _evaluate_item(self, item: dict) -> EvalItemResult:
        query = item["gim_query"]
        query_obj = Query(query)
        try:
            result, gen_time, ttft = self.model.generate_with_timing(query)
        except Exception as e:
            logger.error(f"Error generating result for query '{query}': {e}")
            return EvalItemResult(
                query=query,
                result="Generation Error",
                tags=query_obj.tags[:],
                regex_matched_ids=[],
                num_tags=len(query_obj.tags),
                num_has_prediction=0,
                num_regex=sum(1 for tag in query_obj.tags if tag.regex),
                num_regex_match=0,
                error_msg=str(e),
            )
        regex_matched_ids = []
        for idx, tag in enumerate(result.tags):
            if tag.regex and tag.content and re.fullmatch(tag.regex, tag.content) is not None:
                regex_matched_ids.append(idx)
        result_text = str(result)
        response_tokens = self._count_tokens(result_text)
        throughput = response_tokens / gen_time if gen_time > 0 and response_tokens >= 0 else -1.0
        return EvalItemResult(
            query=query,
            result=result_text,
            tags=result.tags[:],
            regex_matched_ids=regex_matched_ids,
            num_tags=len(result.tags),
            num_has_prediction=sum(1 for tag in result.tags if tag.content),
            num_regex=sum(1 for tag in result.tags if tag.regex),
            num_regex_match=len(regex_matched_ids),
            response_tokens=response_tokens,
            generation_time=gen_time,
            throughput=throughput,
            ttft=ttft,
            error_msg="",
        )

    def evaluate(self) -> EvalResult:
        logger.info(f"Starting evaluation with config: {self.args}")
        total = len(self.dataset) if self.args.first_n == -1 else min(self.args.first_n, len(self.dataset))

        evaled_items = []
        for idx in tqdm(range(total), desc=f"Evaluating {self.args.model_name}"):
            result = self._evaluate_item(self.dataset[idx])
            evaled_items.append(result)

            self._log_progress(total, idx)

        self.end_time = datetime.now()
        logger.info(f"Evaluation completed at {self.end_time}")

        non_error_items = self._filter_non_error_items(evaled_items)
        total_tags = sum(item.num_tags for item in evaled_items)
        valid_tags = sum(item.num_tags for item in non_error_items)
        total_has_prediction = sum(item.num_has_prediction for item in non_error_items)

        total_regex = sum(item.num_regex for item in evaled_items)
        valid_regex = sum(item.num_regex for item in non_error_items)
        total_regex_match = sum(item.num_regex_match for item in non_error_items)
        return EvalResult(
            args=self.args,
            start_time=self.start_time,
            end_time=self.end_time,
            elapsed_minutes=(self.end_time - self.start_time).total_seconds() / 60.0,
            total_queries=total,
            errors=sum(1 for item in evaled_items if item.error_msg),
            total_tags=total_tags,
            valid_tags=valid_tags,
            total_has_prediction=total_has_prediction,
            total_regex=total_regex,
            valid_regex=valid_regex,
            total_regex_match=total_regex_match,
            prediction_rate=total_has_prediction / valid_tags if valid_tags > 0 else 0.0,
            match_rate=total_regex_match / valid_regex if valid_regex > 0 else 0.0,
            avg_response_tokens=self._safe_average(evaled_items, "response_tokens"),
            avg_generation_time=self._safe_average(evaled_items, "generation_time"),
            avg_throughput=self._safe_average(evaled_items, "throughput"),
            avg_ttft=self._safe_average(evaled_items, "ttft"),
            evaled_items=evaled_items,
        )


def conduct_eval(args: Namespace, dataset: Dataset):
    evaluator = MatchEvaluator(args, dataset)
    eval_results = evaluator.evaluate()
    eval_results.dump()
