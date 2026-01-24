"""计算一些基于ppl的评测指标

一些缩写的含义

norm ppl: normalized perplexity, 对原始的ppl进行归一化并缩放后的值
inp: infilling norm ppl, 对填空部分的norm ppl
wnp: windowed norm ppl, 对某个窗口内的norm ppl
l/r wnp: left/right windowed norm ppl, 对填空部分左/右窗口内的norm ppl
"""

from argparse import Namespace
from datetime import datetime
from typing import Literal

import torch

from datasets import Dataset
from gimkit.contexts import Query, Result
from gimkit.schemas import MaskedTag
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from gimbench.base import BaseEvalResult, BaseEvaluator
from gimbench.log import get_logger
from gimbench.models import SimpleGIM


logger = get_logger(__name__)


class EvalItemResult(BaseModel):
    query: str = ""
    result: str = ""

    lwnp: float = -1.0
    rwnp: float = -1.0
    wnp: float = -1.0  # average of left/right
    inp: float = -1.0

    query_tags: int = -1
    result_tags: int = -1
    infilling_ratio: float = -1.0

    query_len: int = -1
    response_len: int = -1
    text_span_and_norm_ppl: dict = {}

    error_msg: str = ""


class EvalResult(BaseEvalResult):
    evaluator_type: Literal["ppl"] = "ppl"

    total: int
    evaluates: int
    errors: int

    avg_lwnp: float = 0.0
    avg_rwnp: float = 0.0
    avg_wnp: float = 0.0
    avg_inp: float = 0.0

    avg_query_tags: float = 0.0
    avg_result_tags: float = 0.0
    avg_infilling_ratio: float = 0.0

    avg_query_len: float = 0.0
    avg_response_len: float = 0.0

    evaled_items: list[EvalItemResult]


class PPLEvaluator(BaseEvaluator):
    def __init__(self, args: Namespace, dataset: Dataset):
        if "gim_query" not in dataset.column_names:
            raise ValueError("Dataset must contain 'gim_query' column for PPL evaluation.")

        super().__init__(args, dataset)

        # SimpleGIM is firstly initialized here to avoid
        # CUDA context contamination in multiprocessing
        self.model = SimpleGIM(args)

        self.ref_model = AutoModelForCausalLM.from_pretrained(args.ref_model_name).to(self.args.ref_model_device)
        self.ref_tokenizer = AutoTokenizer.from_pretrained(args.ref_model_name)

    def _model_call(self, query: str) -> Result:
        """Call the model with the given query and return the Result object."""
        return self.model.generate(query)

    @torch.no_grad()
    def _compute_ppl(self, input_ids: torch.Tensor) -> float:
        outputs = self.ref_model(input_ids, labels=input_ids)
        loss = outputs.loss
        return torch.exp(loss).item()

    def _compute_norm_ppl(self, input_ids: torch.Tensor) -> float:
        """Compute normalized perplexity for the given input IDs."""
        ppl = self._compute_ppl(input_ids)
        norm_ppl = (ppl / self.args.base_model_vocab_size) ** self.args.norm_ppl_alpha
        return norm_ppl

    def _compute_norm_ppl_from_text(self, text: str) -> float:
        enc = self.ref_tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc.input_ids.to(self.args.ref_model_device)
        return self._compute_norm_ppl(input_ids) if input_ids.numel() > 1 else -1.0

    def _evaluate_item(self, item: dict) -> EvalItemResult:
        result_text = "ERROR"
        error_msg = ""
        inp = lwnp = rwnp = wnp = -1.0
        query_text = str(Query(item["gim_query"]))
        span_and_norm_ppl = {
            # "m_0": { "content": xxx
            #          "left_window_text": xxx
            #          "right_window_text": xxx
            #          "content_norm_ppl": x
            #          "left_window_norm_ppl": x
            #          "right_window_norm_ppl": x }
        }
        try:
            result_obj = self._model_call(query_text)
            result_text = str(result_obj)

            _contents = [part if isinstance(part, str) else part.content for part in result_obj.parts]
            _contents = [part if part else "" for part in _contents]
            _start_positions = [len("".join(_contents[:i])) for i in range(len(_contents))]
            content_offset_mapping = [
                (start, start + len(content)) for start, content in zip(_start_positions, _contents, strict=True)
            ]

            token_offset_mapping = self.ref_tokenizer(
                result_text,
                return_offsets_mapping=True,
                add_special_tokens=False,
            ).offset_mapping
            num_tokens = len(token_offset_mapping)

            for part_idx, part in enumerate(result_obj.parts):
                if not isinstance(part, MaskedTag):
                    continue

                content = part.content
                content_start, content_end = content_offset_mapping[part_idx]

                # find first token that contains content_start
                left_window_middle_token_idx = -1
                for token_idx, (token_start, token_end) in enumerate(token_offset_mapping):
                    if token_start <= content_start < token_end:
                        left_window_middle_token_idx = token_idx
                        break
                left_window = (
                    left_window_middle_token_idx - self.args.ppl_window_k,
                    left_window_middle_token_idx + self.args.ppl_window_k,
                )
                if 0 <= left_window[0] < left_window[1] <= num_tokens:
                    left_window_start_pos = token_offset_mapping[left_window[0]][0]
                    left_window_end_pos = token_offset_mapping[left_window[1] - 1][1]
                    left_window_text = result_text[left_window_start_pos:left_window_end_pos]
                else:
                    left_window_text = ""

                # find last token that contains content_end
                right_window_middle_token_idx = -1
                for token_idx in reversed(range(len(token_offset_mapping))):
                    token_start, token_end = token_offset_mapping[token_idx]
                    if token_start < content_end <= token_end:
                        right_window_middle_token_idx = token_idx
                        break
                right_window = (
                    right_window_middle_token_idx - self.args.ppl_window_k,
                    right_window_middle_token_idx + self.args.ppl_window_k,
                )
                if 0 <= right_window[0] < right_window[1] <= num_tokens:
                    right_window_start_pos = token_offset_mapping[right_window[0]][0]
                    right_window_end_pos = token_offset_mapping[right_window[1] - 1][1]
                    right_window_text = result_text[right_window_start_pos:right_window_end_pos]
                else:
                    right_window_text = ""

                span_and_norm_ppl[f"m_{part.id}"] = {
                    "content": content,
                    "left_window_text": left_window_text,
                    "right_window_text": right_window_text,
                }

            for span_key in span_and_norm_ppl:
                # compute content norm ppl
                content = span_and_norm_ppl[span_key]["content"]
                content_norm_ppl = self._compute_norm_ppl_from_text(content) if content else -1.0
                span_and_norm_ppl[span_key]["content_norm_ppl"] = content_norm_ppl

                # compute left window norm ppl
                left_window_text = span_and_norm_ppl[span_key]["left_window_text"]
                left_window_norm_ppl = self._compute_norm_ppl_from_text(left_window_text) if left_window_text else -1.0
                span_and_norm_ppl[span_key]["left_window_norm_ppl"] = left_window_norm_ppl

                # compute right window norm ppl
                right_window_text = span_and_norm_ppl[span_key]["right_window_text"]
                if right_window_text:
                    right_window_norm_ppl = self._compute_norm_ppl_from_text(right_window_text)
                else:
                    right_window_norm_ppl = -1.0
                span_and_norm_ppl[span_key]["right_window_norm_ppl"] = right_window_norm_ppl

            valid_inps = [
                span_and_norm_ppl[span_key]["content_norm_ppl"]
                for span_key in span_and_norm_ppl
                if span_and_norm_ppl[span_key]["content_norm_ppl"] >= 0
            ]
            valid_lwnps = [
                span_and_norm_ppl[span_key]["left_window_norm_ppl"]
                for span_key in span_and_norm_ppl
                if span_and_norm_ppl[span_key]["left_window_norm_ppl"] >= 0
            ]
            valid_rwnps = [
                span_and_norm_ppl[span_key]["right_window_norm_ppl"]
                for span_key in span_and_norm_ppl
                if span_and_norm_ppl[span_key]["right_window_norm_ppl"] >= 0
            ]

            inp = sum(valid_inps) / len(valid_inps) if valid_inps else -1.0
            lwnp = sum(valid_lwnps) / len(valid_lwnps) if valid_lwnps else -1.0
            rwnp = sum(valid_rwnps) / len(valid_rwnps) if valid_rwnps else -1.0
            wnp = (lwnp + rwnp) / 2.0 if lwnp >= 0 and rwnp >= 0 else -1.0

        except Exception as e:
            error_msg = repr(e)
            logger.exception(error_msg)

        return EvalItemResult(
            query=query_text,
            result=result_text,
            lwnp=lwnp,
            rwnp=rwnp,
            wnp=wnp,
            inp=inp,
            query_tags=len(Query(query_text).tags),
            result_tags=len(Result(result_text).tags),
            infilling_ratio=(1 - len(Result(result_text).tags) / len(Query(query_text).tags))
            if len(Query(query_text).tags) > 0
            else -1.0,
            query_len=len(query_text),
            response_len=len(result_text),
            text_span_and_norm_ppl=span_and_norm_ppl,
            error_msg=error_msg,
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

        return EvalResult(
            total=total,
            evaluates=len(evaled_items),
            errors=sum(1 for item in evaled_items if item.error_msg),
            avg_lwnp=self._safe_average(evaled_items, "lwnp"),
            avg_rwnp=self._safe_average(evaled_items, "rwnp"),
            avg_wnp=self._safe_average(evaled_items, "wnp"),
            avg_inp=self._safe_average(evaled_items, "inp"),
            avg_query_tags=self._safe_average(evaled_items, "query_tags"),
            avg_result_tags=self._safe_average(evaled_items, "result_tags"),
            avg_infilling_ratio=self._safe_average(evaled_items, "infilling_ratio"),
            avg_query_len=self._safe_average(evaled_items, "query_len"),
            avg_response_len=self._safe_average(evaled_items, "response_len"),
            start_time=self.start_time,
            end_time=self.end_time,
            elapsed_minutes=(self.end_time - self.start_time).total_seconds() / 60.0,
            args=self.args,
            evaled_items=evaled_items,
        )


def conduct_eval(args: Namespace, ds: Dataset):
    if args.no_gimkit:
        raise NotImplementedError("Only GIM evaluation is implemented in this evaluator.")
    evaluator = PPLEvaluator(args, ds)
    result = evaluator.evaluate()
    result.dump()
