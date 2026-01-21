import subprocess
import sys

from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import gimkit

from datasets import Dataset
from pydantic import BaseModel, Field, field_serializer

import gimbench

from gimbench.arguments import SECRET_ARGS
from gimbench.log import get_logger


logger = get_logger(__name__)


def _create_eval_env() -> dict[str, str]:
    try:
        git_repo = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).strip().decode("utf-8")
        git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode("utf-8")
        git_commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        git_repo = "unknown"
        git_branch = "unknown"
        git_commit_id = "unknown"
    return {
        "exec_command": " ".join([sys.executable, *sys.argv]),
        "gimbench_version": gimbench.__version__,
        "gimbench_file": str(gimbench.__file__),
        "gimkit_version": gimkit.__version__,
        "gimkit_file": str(gimkit.__file__),
        "git_repo": git_repo,
        "git_branch": git_branch,
        "git_commit_id": git_commit_id,
    }


class BaseEvalResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    eval_env: dict[str, str] = Field(default_factory=_create_eval_env)

    evaluator_type: Literal["mcqa", "ctp", "match"]

    args: Namespace

    start_time: datetime
    end_time: datetime
    elapsed_minutes: float

    evaled_items: list

    @field_serializer("args")
    def serialize_args(self, value: Namespace) -> dict[str, Any]:
        serialized = vars(value).copy()
        for secret_arg in SECRET_ARGS:
            if serialized.get(secret_arg):
                serialized[secret_arg] = "****"
        return serialized

    def dump(self, filepath: str | None = None) -> None:
        if filepath is None:
            dataset = getattr(self.args, "dataset", {})
            dataset_path = dataset.get("path", "unknown_dataset") if isinstance(dataset, dict) else "unknown_dataset"
            model_name = getattr(self.args, "model_name", "unknown_model")
            filename = f"{model_name}_{dataset_path}_{self.start_time.strftime('%y%m%d-%H%M%S')}.json".replace("/", "_")
            filepath = str(Path(self.args.output_dir or ".") / filename)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            f.write(self.model_dump_json(indent=4))
        logger.info(f"Saved evaluation results to {filepath}")


class BaseEvaluator:
    def __init__(self, args: Namespace, dataset: Dataset):
        self.start_time = datetime.now()
        self.dataset = dataset
        self.args = args

    @staticmethod
    def _safe_average(items: list, attr: str) -> float:
        values = [getattr(item, attr) for item in items if getattr(item, attr) != -1]
        return sum(values) / len(values) if values else 0.0

    def _log_progress(self, total: int, curr_idx: int, log_interval: int = 10) -> None:
        completed = curr_idx + 1
        if completed % log_interval == 0:
            speed = (datetime.now() - self.start_time).total_seconds() / completed
            logger.info(
                f"Progress: {completed}/{total} items evaluated with speed {speed:.2f} seconds/item. "
                f"Time Remaining: {(total - completed) * speed / 60:.2f} minutes"
            )
