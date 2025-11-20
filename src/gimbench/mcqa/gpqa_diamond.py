# https://huggingface.co/datasets/Idavidrein/gpqa

import random

from datasets import load_dataset

from gimbench.arguments import get_args
from gimbench.log import get_logger
from gimbench.mcqa.evaluators import conduct_eval


logger = get_logger(__name__)


def _format_gpqa(example: dict, seed: int) -> dict:
    question = example["Question"].strip()
    answers = [
        example["Correct Answer"].strip(),
        example["Incorrect Answer 1"].strip(),
        example["Incorrect Answer 2"].strip(),
        example["Incorrect Answer 3"].strip(),
    ]
    indices = list(range(len(answers)))
    random.seed(seed + hash(question))
    random.shuffle(indices)

    question_with_answer_options = f"{question}\n\nChoices:\n"
    for i, idx in enumerate(indices):
        question_with_answer_options += f"{chr(ord('A') + i)}. {answers[idx]}\n"

    letter_choices = [chr(ord("A") + i) for i in range(len(answers))]
    correct_choice = chr(ord("A") + indices.index(0))

    return {
        "question": question_with_answer_options,
        "choices": letter_choices,
        "correct_choice": correct_choice,
    }


if __name__ == "__main__":
    args = get_args()
    args.dataset = {"path": "Idavidrein/gpqa", "name": "gpqa_diamond", "split": "train"}

    ds = load_dataset(args.dataset["path"], args.dataset["name"], split=args.dataset["split"]).map(
        lambda x: _format_gpqa(x, seed=args.seed)
    )
    logger.info(f"Loaded {len(ds)} samples from dataset {args.dataset}")
    logger.info(f"First sample: {ds[0]}")

    conduct_eval(args, ds)
