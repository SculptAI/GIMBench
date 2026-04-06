# https://huggingface.co/datasets/loubnabnl/humaneval_infilling

from datasets import concatenate_datasets, load_dataset

from gimbench.arguments import get_args
from gimbench.code.evaluators import conduct_eval
from gimbench.log import get_logger


logger = get_logger(__name__)

_SUBSETS = ["single_line", "multi_line", "random_span"]


if __name__ == "__main__":
    args = get_args()
    args.dataset = {
        "path": "loubnabnl/humaneval_infilling",
        "name": _SUBSETS,
        "split": "test",
    }

    datasets_list = []
    for subset in _SUBSETS:
        sub_ds = load_dataset(args.dataset["path"], subset, split=args.dataset["split"])
        sub_ds = sub_ds.add_column("subset", [subset] * len(sub_ds))
        datasets_list.append(sub_ds)

    ds = concatenate_datasets(datasets_list)
    logger.info(f"Loaded {len(ds)} samples from dataset {args.dataset}")
    logger.info(f"Columns: {ds.column_names}")
    logger.info(f"First sample: {ds[0]}")

    conduct_eval(args, ds)
