# https://huggingface.co/datasets/allenai/qasc

import random

from datasets import load_dataset

from gimbench.arguments import get_args
from gimbench.log import get_logger
from gimbench.mcqa.evaluators import conduct_eval


logger = get_logger(__name__)


def _format_qasc(example: dict, seed: int) -> dict:
    question = example["question"].strip()
    answers = example["choices"]["text"]
    indices = list(range(len(answers)))
    random.seed(seed + hash(question))
    random.shuffle(indices)

    question_with_answer_options = f"{question}\n\nChoices:\n"
    for i, idx in enumerate(indices):
        question_with_answer_options += f"{chr(ord('A') + i)}. {answers[idx]}\n"

    letter_choices = [chr(ord("A") + i) for i in range(len(answers))]
    correct_choice = chr(ord("A") + indices.index(ord(example["answerKey"]) - ord("A")))

    return {
        "question": question_with_answer_options,
        "choices": letter_choices,
        "correct_choice": correct_choice,
    }


if __name__ == "__main__":
    args = get_args()
    args.dataset = {"path": "allenai/qasc", "name": None, "split": "validation"}

    ds = load_dataset(args.dataset["path"], args.dataset["name"], split=args.dataset["split"]).map(
        lambda x: _format_qasc(x, seed=args.seed)
    )
    logger.info(f"Loaded {len(ds)} samples from dataset {args.dataset}")
    logger.info(f"First sample: {ds[0]}")

    conduct_eval(args, ds)
