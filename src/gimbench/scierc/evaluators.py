import json
import re

from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Literal

from datasets import Dataset
from gimkit import guide
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from gimbench.base import BaseEvalResult, BaseEvaluator
from gimbench.log import get_logger
from gimbench.models import SimpleCommon, SimpleGIM


logger = get_logger(__name__)

RELATION_TYPES = {
    "USED-FOR",
    "FEATURE-OF",
    "PART-OF",
    "CONJUNCTION",
    "EVALUATE-FOR",
    "HYPONYM-OF",
    "COMPARE",
}

ENTITY_TYPES = {
    "Task",
    "Method",
    "Metric",
    "Material",
    "OtherScientificTerm",
    "Generic",
}


class EvalItemResult(BaseModel):
    doc_key: str = ""
    query: str = ""
    response: str = ""

    gold_relations: int = 0
    predicted_relations: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    query_tokens: int = -1
    response_tokens: int = -1
    generation_time: float = -1.0
    throughput: float = -1.0
    ttft: float = -1.0

    error_msg: str = ""


class EvalResult(BaseEvalResult):
    evaluator_type: Literal["scierc"] = "scierc"

    total_docs: int
    evaluates: int
    errors: int

    total_gold_relations: int
    total_predicted_relations: int
    true_positives: int
    false_positives: int
    false_negatives: int

    precision: float
    recall: float
    f1: float

    avg_query_tokens: float
    avg_response_tokens: float
    avg_generation_time: float
    avg_throughput: float
    avg_ttft: float

    evaled_items: list[EvalItemResult]


def _canonical_relation_label(label: str) -> str:
    return label.strip().upper().replace("_", "-").replace(" ", "-")


def _canonical_entity_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    text = text.strip()

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "[":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            continue

    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("Model output must be a JSON array.")
    return parsed


class SciERCEvaluator(BaseEvaluator):
    def __init__(self, args: Namespace, dataset: Dataset):
        super().__init__(args, dataset)
        self._counter_tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(args.counter_tokenizer)
        logger.info(f"Loaded tokenizer {args.counter_tokenizer} for token counting.")

    def _count_tokens(self, text: str) -> int:
        return len(self._counter_tokenizer.encode(text))

    def _form_query(self, item: dict) -> str:
        relation_types = ", ".join(sorted(RELATION_TYPES))
        entity_types = ", ".join(sorted(ENTITY_TYPES))

        return (
            "Extract scientific relations from the abstract below (SciERC style).\n"
            "Return ONLY a JSON array. Do not include extra text.\n"
            "Each element must be an object with keys: head, head_type, relation, tail, tail_type.\n"
            f"Allowed relation labels: {relation_types}.\n"
            f"Allowed entity labels: {entity_types}.\n"
            "Preserve relation direction as expressed in the text.\n"
            "Do not invent entities that are not explicitly mentioned.\n"
            f"Return at most {self.args.scierc_max_relations} relations.\n\n"
            "Abstract:\n"
            f"{item['text']}"
        )

    def _parse_predicted_relations(self, response: str) -> set[tuple[str, str, str]]:
        rows = _extract_json_array(response)
        triples = set()

        for row in rows:
            if not isinstance(row, dict):
                continue
            head = str(row.get("head", "")).strip()
            tail = str(row.get("tail", "")).strip()
            relation = _canonical_relation_label(str(row.get("relation", "")))

            if not head or not tail or relation not in RELATION_TYPES:
                continue

            triples.add((_canonical_entity_text(head), relation, _canonical_entity_text(tail)))

        return triples

    def _gold_relations(self, item: dict) -> set[tuple[str, str, str]]:
        triples = set()
        for rel in item["gold_relations"]:
            triples.add(
                (
                    _canonical_entity_text(rel["head"]),
                    _canonical_relation_label(rel["relation"]),
                    _canonical_entity_text(rel["tail"]),
                )
            )
        return triples

    def _evaluate_item(self, item: dict) -> EvalItemResult:
        query = self._form_query(item)
        response = ""
        gen_time = -1.0
        ttft = -1.0

        try:
            if isinstance(self.model, SimpleGIM) and self.args.record_timing:
                result, gen_time, ttft = self.model.generate_with_timing(query)
                response = str(result)
            elif isinstance(self.model, SimpleGIM):
                response = str(self.model.generate(query))
            else:
                response = self.model.generate(query)

            gold = self._gold_relations(item)
            pred = self._parse_predicted_relations(response)

            tp = len(gold & pred)
            fp = len(pred - gold)
            fn = len(gold - pred)
            error_msg = ""
        except Exception as e:
            logger.exception("SciERC item evaluation failed")
            gold = self._gold_relations(item)
            pred = set()
            tp = 0
            fp = 0
            fn = len(gold)
            error_msg = str(e)

        response_tokens = self._count_tokens(response) if response else -1
        throughput = response_tokens / gen_time if gen_time > 0 and response_tokens >= 0 else -1.0
        return EvalItemResult(
            doc_key=item["doc_key"],
            query=query,
            response=response,
            gold_relations=len(gold),
            predicted_relations=len(pred),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            query_tokens=self._count_tokens(query),
            response_tokens=response_tokens,
            generation_time=gen_time,
            throughput=throughput,
            ttft=ttft,
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

        non_error_items = self._filter_non_error_items(evaled_items)
        errors = sum(1 for item in evaled_items if item.error_msg)
        tp = sum(item.true_positives for item in non_error_items)
        fp = sum(item.false_positives for item in non_error_items)
        fn = sum(item.false_negatives for item in non_error_items)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        self.end_time = datetime.now()
        logger.info(
            f"SciERC micro scores over {total} docs: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, errors={errors}"
        )

        return EvalResult(
            total_docs=total,
            evaluates=len(evaled_items),
            errors=errors,
            total_gold_relations=sum(item.gold_relations for item in non_error_items),
            total_predicted_relations=sum(item.predicted_relations for item in non_error_items),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            avg_query_tokens=self._safe_average(evaled_items, "query_tokens"),
            avg_response_tokens=self._safe_average(evaled_items, "response_tokens"),
            avg_generation_time=self._safe_average(evaled_items, "generation_time"),
            avg_throughput=self._safe_average(evaled_items, "throughput"),
            avg_ttft=self._safe_average(evaled_items, "ttft"),
            start_time=self.start_time,
            end_time=self.end_time,
            elapsed_minutes=(self.end_time - self.start_time).total_seconds() / 60.0,
            args=self.args,
            evaled_items=evaled_items,
        )


class GIMSciERCEvaluator(SciERCEvaluator):
    def __init__(self, args: Namespace, dataset: Dataset):
        super().__init__(args, dataset)
        self.model = SimpleGIM(args)

    def _form_query(self, item: dict) -> str:
        relation_types = ", ".join(sorted(RELATION_TYPES))
        entity_types = ", ".join(sorted(ENTITY_TYPES))

        prompt = (
            "Extract scientific relations from the abstract below (SciERC style).\n"
            f"Allowed relation labels: {relation_types}.\n"
            f"Allowed entity labels: {entity_types}.\n"
            "Preserve relation direction as expressed in the text.\n"
            "Do not invent entities that are not explicitly mentioned.\n"
            f"Extract up to {self.args.scierc_max_relations} relations. If there are fewer, leave the remaining fields empty or use 'None'.\n\n"
            "Abstract:\n"
            f"{item['text']}\n\n"
            "## Extracted Relations\n\n"
        )
        for i in range(self.args.scierc_max_relations):
            prompt += f"- Relation {i+1}:\n"
            prompt += f"  - Head Entity: {guide(name=f'head_{i}')}\n"
            prompt += f"  - Relation Type: {guide(name=f'relation_{i}')}\n"
            prompt += f"  - Tail Entity: {guide(name=f'tail_{i}')}\n\n"

        return prompt

    def _evaluate_item(self, item: dict) -> EvalItemResult:
        query = self._form_query(item)
        response = ""
        gen_time = -1.0
        ttft = -1.0

        try:
            if self.args.record_timing:
                result, gen_time, ttft = self.model.generate_with_timing(query)
            else:
                result = self.model.generate(query)
            response = str(result)

            gold = self._gold_relations(item)
            pred = set()

            for i in range(self.args.scierc_max_relations):
                head = result.tags[f"head_{i}"].content
                rel = result.tags[f"relation_{i}"].content
                tail = result.tags[f"tail_{i}"].content

                if head and rel and tail:
                    head = _canonical_entity_text(head)
                    rel = _canonical_relation_label(rel)
                    tail = _canonical_entity_text(tail)

                    if head and tail and head != "none" and tail != "none" and rel in RELATION_TYPES:
                        pred.add((head, rel, tail))

            tp = len(gold & pred)
            fp = len(pred - gold)
            fn = len(gold - pred)
            error_msg = ""
        except Exception as e:
            logger.exception("SciERC item evaluation failed")
            gold = self._gold_relations(item)
            pred = set()
            tp = 0
            fp = 0
            fn = len(gold)
            error_msg = str(e)

        response_tokens = self._count_tokens(response) if response else -1
        throughput = response_tokens / gen_time if gen_time > 0 and response_tokens >= 0 else -1.0
        return EvalItemResult(
            doc_key=item["doc_key"],
            query=query,
            response=response,
            gold_relations=len(gold),
            predicted_relations=len(pred),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            query_tokens=self._count_tokens(query),
            response_tokens=response_tokens,
            generation_time=gen_time,
            throughput=throughput,
            ttft=ttft,
            error_msg=error_msg,
        )


class CommonSciERCEvaluator(SciERCEvaluator):
    def __init__(self, args: Namespace, dataset: Dataset):
        super().__init__(args, dataset)
        self.model = SimpleCommon(args)


def conduct_eval(args: Namespace, ds: Dataset) -> None:
    evaluator = CommonSciERCEvaluator(args, ds) if args.no_gimkit else GIMSciERCEvaluator(args, ds)
    result = evaluator.evaluate()
    result.dump()
