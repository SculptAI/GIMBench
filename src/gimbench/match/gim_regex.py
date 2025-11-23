# https://huggingface.co/datasets/Sculpt-AI/GIMBench-regex-match


from datasets import load_dataset

from gimbench.arguments import get_args
from gimbench.log import get_logger
from gimbench.match.evaluators import conduct_eval


logger = get_logger(__name__)

if __name__ == "__main__":
    args = get_args()
    args.dataset = {"path": "Sculpt-AI/GIMBench-regex-match", "split": "test"}

    ds = load_dataset(args.dataset["path"], split=args.dataset["split"])
    logger.info(f"Loaded {len(ds)} samples from dataset {args.dataset}")
    logger.info(f"First sample: {ds[0]}")

    conduct_eval(args, ds)
