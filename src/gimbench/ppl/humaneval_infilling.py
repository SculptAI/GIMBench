# https://huggingface.co/datasets/loubnabnl/humaneval_infilling

from datasets import concatenate_datasets, load_dataset
from gimkit.schemas import MaskedTag

from gimbench.arguments import get_args
from gimbench.log import get_logger
from gimbench.ppl.evaluators import conduct_eval


logger = get_logger(__name__)

_INFILL_MASK = MaskedTag(name="infill", desc="Fill in the missing code")

_SUBSETS = ["single_line", "multi_line", "random_span"]


def _format_humaneval_infilling(example: dict) -> dict:
    gim_query = f"{example['prompt']}{_INFILL_MASK}{example['suffix']}"
    gim_response = str(MaskedTag(name="infill", content=example["canonical_solution"]))
    return {"gim_query": gim_query, "gim_response": gim_response}


if __name__ == "__main__":
    args = get_args()
    args.dataset = {
        "path": "loubnabnl/humaneval_infilling",
        "name": _SUBSETS,
        "split": "test",
    }

    ds = concatenate_datasets(
        [
            load_dataset(args.dataset["path"], subset, split=args.dataset["split"]).map(_format_humaneval_infilling)
            for subset in args.dataset["name"]
        ]
    ).select_columns(["gim_query", "gim_response"])
    logger.info(f"Loaded {len(ds)} samples from dataset {args.dataset}")
    logger.info(f"First sample: {ds[0]}")

    conduct_eval(args, ds)
