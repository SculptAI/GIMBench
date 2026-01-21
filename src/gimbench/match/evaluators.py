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

    error_msg: str = ""


class EvalResult(BaseEvalResult):
    evaluator_type: Literal["match"] = "match"

    total_queries: int
    errors: int

    total_tags: int
    total_has_prediction: int
    total_regex: int
    total_regex_match: int

    prediction_rate: float
    match_rate: float

    evaled_items: list[EvalItemResult]


class MatchEvaluator(BaseEvaluator):
    def __init__(self, args: Namespace, dataset: Dataset):
        if args.no_gimkit:
            raise ValueError("GIMKit must be enabled for MatchEvaluator.")

        super().__init__(args, dataset)
        self.model = SimpleGIM(args)

    def _evaluate_item(self, item: dict) -> EvalItemResult:
        query = item["gim_query"]
        query_obj = Query(query)
        try:
            result = self.model.generate(query)
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
        return EvalItemResult(
            query=query,
            result=str(result),
            tags=result.tags[:],
            regex_matched_ids=regex_matched_ids,
            num_tags=len(result.tags),
            num_has_prediction=sum(1 for tag in result.tags if tag.content),
            num_regex=sum(1 for tag in result.tags if tag.regex),
            num_regex_match=len(regex_matched_ids),
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

        total_tags = sum(item.num_tags for item in evaled_items)
        total_has_prediction = sum(item.num_has_prediction for item in evaled_items)
        total_regex = sum(item.num_regex for item in evaled_items)
        total_regex_match = sum(item.num_regex_match for item in evaled_items)
        return EvalResult(
            args=self.args,
            start_time=self.start_time,
            end_time=self.end_time,
            elapsed_minutes=(self.end_time - self.start_time).total_seconds() / 60.0,
            total_queries=total,
            errors=sum(1 for item in evaled_items if item.error_msg),
            total_tags=total_tags,
            total_has_prediction=total_has_prediction,
            total_regex=total_regex,
            total_regex_match=total_regex_match,
            prediction_rate=total_has_prediction / total_tags if total_tags > 0 else 0.0,
            match_rate=total_regex_match / total_regex if total_regex > 0 else 0.0,
            evaled_items=evaled_items,
        )

    @staticmethod
    def print_beautiful_stats(eval_results: EvalResult) -> None:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        console = Console()

        args = eval_results.args
        info_text = Text.from_markup(
            f"[bold]Model:[/bold] [cyan]{args.model_name}[/cyan] "
            f"[bold]GIM Prompt:[/bold] [green]{args.use_gim_prompt}[/green] "
            f"[bold]Output Type:[/bold] [magenta]{args.output_type}[/magenta]"
        )
        console.print(Panel(info_text, title="Run Arguments", border_style="blue", expand=False))

        table = Table()
        table.add_column("Tags", justify="right", style="magenta")
        table.add_column("Predicted", justify="right", style="green")
        table.add_column("Regex", justify="right", style="blue")
        table.add_column("Matched", justify="right", style="yellow")
        table.add_column("Prediction Rate", justify="right", style="green")
        table.add_column("Match Rate", justify="right", style="yellow")
        for result in eval_results.evaled_items:
            pred_rate = f"{result.num_has_prediction / result.num_tags:.2%}" if result.num_tags > 0 else "N/A"
            match_rate = f"{result.num_regex_match / result.num_regex:.2%}" if result.num_regex > 0 else "N/A"

            table.add_row(
                str(result.num_tags),
                str(result.num_has_prediction),
                str(result.num_regex),
                str(result.num_regex_match),
                pred_rate,
                match_rate,
            )

        console.print(table)


def conduct_eval(args: Namespace, dataset: Dataset):
    evaluator = MatchEvaluator(args, dataset)
    eval_results = evaluator.evaluate()
    MatchEvaluator.print_beautiful_stats(eval_results)
    eval_results.dump()
