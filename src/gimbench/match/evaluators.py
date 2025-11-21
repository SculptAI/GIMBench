import argparse
import logging
import re

from datetime import datetime
from typing import Literal

from datasets import Dataset
from gimkit.contexts import Query
from gimkit.schemas import MaskedTag
from pydantic import BaseModel

from gimbench.base import BaseEvalResult
from gimbench.models import SimpleGIM


logging.getLogger("gimkit").setLevel(logging.DEBUG)


class EvalTag(BaseModel):
    tag: MaskedTag
    has_prediction: bool
    has_regex: bool
    regex_match: bool


class EvalQuery(BaseModel):
    query: str
    result: str
    eval_tags: list[EvalTag]
    num_tags: int
    num_has_prediction: int
    num_regex: int
    num_regex_match: int


class EvalResult(BaseEvalResult):
    evaluator_type: Literal["match"] = "match"

    evaled_items: list[EvalQuery]


def conduct_eval(args: argparse.Namespace, dataset: Dataset):
    start_time = datetime.now()
    logging.getLogger("gimkit").setLevel(logging.DEBUG)

    model = SimpleGIM(args)
    queries = dataset["gim_query"][: args.first_n if args.first_n > 0 else None]

    eval_results = []
    for query in queries:
        try:
            result = model.generate(query)
        except Exception as e:
            print(f"Skipping query '{query}' due to: {e}")
            query_obj = Query(query)
            eval_results.append(
                EvalQuery(
                    query=query,
                    result="Invalid Format Error",
                    eval_tags=[],
                    num_tags=len(query_obj.tags),
                    num_has_prediction=0,
                    num_regex=sum(1 for tag in query_obj.tags if tag.regex is not None),
                    num_regex_match=0,
                )
            )
            continue
        eval_items = []
        for tag in result.tags:
            has_prediction = False
            has_regex = False
            regex_match = False
            if tag.content:
                has_prediction = True
            if tag.regex:
                has_regex = True
            if tag.content and tag.regex and re.fullmatch(tag.regex, tag.content) is not None:
                regex_match = True
            eval_items.append(
                EvalTag(tag=tag, has_prediction=has_prediction, has_regex=has_regex, regex_match=regex_match)
            )
        eval_result = EvalQuery(
            query=query,
            result=str(result),
            eval_tags=eval_items,
            num_tags=len(result.tags),
            num_has_prediction=sum(1 for item in eval_items if item.has_prediction),
            num_regex=sum(1 for item in eval_items if item.has_regex),
            num_regex_match=sum(1 for item in eval_items if item.regex_match),
        )
        eval_results.append(eval_result)

    end_time = datetime.now()
    eval_results_obj = EvalResult(
        args=args,
        start_time=start_time,
        end_time=end_time,
        elapsed_minutes=(end_time - start_time).total_seconds() / 60.0,
        evaled_items=eval_results,
    )
    print_beautiful_stats(eval_results_obj)
    eval_results_obj.dump()


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

    total_tags = 0
    total_has_prediction = 0
    total_regex = 0
    total_regex_match = 0

    for result in eval_results.evaled_items:
        total_tags += result.num_tags
        total_has_prediction += result.num_has_prediction
        total_regex += result.num_regex
        total_regex_match += result.num_regex_match

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
