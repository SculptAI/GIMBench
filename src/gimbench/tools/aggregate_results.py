import csv
import json

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from gimbench.log import get_logger


logger = get_logger(__name__)


def to_csv(results, output_file):
    if not results:
        logger.warning("No results to write.")
        return

    keys = results[0].keys()
    with open(output_file, "w", newline="") as output_csv:
        dict_writer = csv.DictWriter(output_csv, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    logger.info(f"Results written to {output_file}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Directory containing evaluation result files.",
    )
    parser.add_argument(
        "--exclude_fields",
        type=str,
        nargs="*",
        default=["evaled_items"],
        help="Fields to exclude from the final results.",
    )
    args = parser.parse_args()

    all_results = defaultdict(list)
    for result_file in Path(args.output_dir).glob("*.json"):
        with open(result_file) as f:
            result = json.load(f)
            # `mcqa` is for backward compatibility
            evaluator_type = result.get("evaluator_type", "mcqa")
            for field in args.exclude_fields:
                if field in result:
                    del result[field]
            result = {"filename": result_file.name} | result
            all_results[evaluator_type].append(result)

    for evaluator_type, results in all_results.items():
        to_csv(results, Path(args.output_dir) / f"aggregated_results_{evaluator_type}.csv")
