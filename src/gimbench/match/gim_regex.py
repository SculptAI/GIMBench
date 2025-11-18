import argparse
import copy
import json
import logging
import os
import re

from dataclasses import dataclass
from typing import Literal, cast

from gimkit import from_openai, from_vllm_offline
from gimkit.contexts import Query, Result
from gimkit.exceptions import InvalidFormatError
from gimkit.models.openai import OpenAI as GIMKitOpenAI
from gimkit.models.vllm_offline import VLLMOffline as GIMKitvLLMOffline
from gimkit.schemas import MaskedTag

from datasets import load_dataset
from gimbench.arguments import get_args
from gimbench.models import SimpleGIM
logging.getLogger("gimkit").setLevel(logging.DEBUG)


@dataclass
class EvalTag:
    tag: MaskedTag
    has_prediction: bool
    has_regex: bool
    regex_match: bool


@dataclass
class EvalQuery:
    query: str
    result: str
    eval_tags: list[EvalTag]
    num_tags: int
    num_has_prediction: int
    num_regex: int
    num_regex_match: int


@dataclass
class EvalResult:
    run_args: argparse.Namespace
    eval_queries: list[EvalQuery]


def conduct_eval(
    queries: list[str], model: SimpleGIM, args: argparse.Namespace
) -> list[EvalQuery]:
    output_type = args.output_type
    if output_type == "none":
        output_type = None

    eval_results = []
    for query in queries:
        try:
            result = model.generate(query)
        except InvalidFormatError as e:
            print(f"Skipping query '{query}' due to invalid format: {e}")
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
            eval_items.append(EvalTag(tag, has_prediction, has_regex, regex_match))
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
    return eval_results


def print_beautiful_stats(eval_results: EvalResult) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()

    run_args = eval_results.run_args
    info_text = Text.from_markup(
        f"[bold]Model:[/bold] [cyan]{run_args.model_name}[/cyan] "
        f"[bold]GIM Prompt:[/bold] [green]{run_args.use_gim_prompt}[/green] "
        f"[bold]Output Type:[/bold] [magenta]{run_args.output_type}[/magenta]"
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

    for result in eval_results.eval_queries:
        total_tags += result.num_tags
        total_has_prediction += result.num_has_prediction
        total_regex += result.num_regex
        total_regex_match += result.num_regex_match

        pred_rate = (
            f"{result.num_has_prediction / result.num_tags:.2%}" if result.num_tags > 0 else "N/A"
        )
        match_rate = (
            f"{result.num_regex_match / result.num_regex:.2%}" if result.num_regex > 0 else "N/A"
        )

        table.add_row(
            str(result.num_tags),
            str(result.num_has_prediction),
            str(result.num_regex),
            str(result.num_regex_match),
            pred_rate,
            match_rate,
        )

    console.print(table)


def save_eval_results(eval_results: EvalResult) -> None:
    eval_results_to_save = copy.deepcopy(eval_results)
    if hasattr(eval_results_to_save.run_args, "api_key"):
        eval_results_to_save.run_args.api_key = "***"

    save_dir = eval_results.run_args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    model_name = re.sub(r"[^\w\-_. ]", "_", eval_results.run_args.model_name)

    save_path = f"{save_dir}/eval_results_{model_name}"
    if eval_results.run_args.use_gim_prompt:
        save_path += "_gim_prompt"
    if eval_results.run_args.output_type and eval_results.run_args.output_type != "none":
        save_path += f"_{eval_results.run_args.output_type}"
    save_path += ".json"

    with open(save_path, "w") as f:
        json.dump(
            eval_results_to_save, f, default=lambda o: o.__dict__, indent=2, ensure_ascii=False
        )


if __name__ == "__main__":
    args = get_args()
    model = SimpleGIM(args)
    dataset = load_dataset("Sculpt-AI/GIMBench-regex-match", split="test")
    queries = dataset["gim_query"][:args.first_n if args.first_n > 0 else None]
    eval_queries = conduct_eval(queries, model, args)
    eval_results = EvalResult(run_args=args, eval_queries=eval_queries)
    print_beautiful_stats(eval_results)
    save_eval_results(eval_results)
