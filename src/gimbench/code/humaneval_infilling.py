# https://huggingface.co/datasets/Sculpt-AI/humaneval_infilling

from datasets import load_dataset, concatenate_datasets

from gimbench.arguments import get_args
from gimbench.code.evaluators import conduct_eval
from gimbench.log import get_logger


logger = get_logger(__name__)


if __name__ == "__main__":
    args = get_args()
    args.dataset = {
        "path": "Sculpt-AI/humaneval_infilling",
        "subsets": [
            "MultiLine",
            "RandomSpan",
            "RandomSpanLight",
            "SingleLine"
        ]
    }

    ds = concatenate_datasets([
        load_dataset(args.dataset["path"], split="test", name=subset)
        for subset in args.dataset["subsets"]
    ]).shuffle(seed=args.seed)
    logger.info(f"Loaded {len(ds)} samples from dataset {args.dataset}")
    logger.info(f"Columns: {ds.column_names}")
    logger.info(f"First sample: {ds[0]}")

    conduct_eval(args, ds)
