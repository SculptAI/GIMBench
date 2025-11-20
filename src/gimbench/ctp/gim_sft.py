# https://huggingface.co/datasets/Sculpt-AI/GIM-SFT

from datasets import concatenate_datasets, load_dataset

from gimbench.arguments import get_args
from gimbench.ctp.evaluators import conduct_eval
from gimbench.log import get_logger


logger = get_logger(__name__)


if __name__ == "__main__":
    args = get_args()
    args.dataset = {
        "path": "Sculpt-AI/GIM-SFT",
        "name": [
            "gsm8k_reasoning",
            "hk_o1aw",
            "lima",
            "o1_journey",
            "process_bench",
            "uhgeval",
            "cnn_daily_mail",
            "magpie_reasoning",
            "kaist_cot",
            "numina_math",
        ],
        "split": "train",
        "max_per_subset": 200,
    }

    ds = (
        concatenate_datasets(
            [
                load_dataset(args.dataset["path"], subset, split=args.dataset["split"]).select(
                    range(args.dataset["max_per_subset"])
                )
                for subset in args.dataset["name"]
            ]
        )
        .shuffle(seed=args.seed)
        .select_columns(["gim_query"])
    )
    logger.info(f"Loaded {len(ds)} samples from dataset {args.dataset}")
    logger.info(f"First sample: {ds[0]}")

    conduct_eval(args, ds)
