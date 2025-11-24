import subprocess

from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from datasets import Dataset
from pydantic import BaseModel, field_serializer

from gimbench.arguments import SECRET_ARGS
from gimbench.log import get_logger


GIT_BRANCH = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode("utf-8")
GIT_COMMIT_ID = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")

logger = get_logger(__name__)


class BaseEvalResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    git_branch: str = GIT_BRANCH
    git_commit_id: str = GIT_COMMIT_ID

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
