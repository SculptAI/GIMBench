import random
import re

from copy import deepcopy
from multiprocessing import Pool, current_process

from datasets import Dataset, load_dataset
from gimkit import guide as g  # noqa: F401
from gimkit.contexts import Query, Response
from gimkit.schemas import MaskedTag
from openai import OpenAI
from tqdm import tqdm


SYS_PROMPT = '''
You are an expert in **synthetic data generation**.

## Background

I am building a benchmark for a **Guided Infilling** task.
For example, given a query:

f"""4 {g(desc="a math operator")} 6 {g(desc="a math operator")} 4 {g(desc="a math operator")} 4 = 24"""

the model should produce a fully infilled response such as:

4 * 6 + 4 - 4 = 24

The infilling is controlled by the function:

```
g(desc: str | None = None, regex: str | None = None)
```

The model must fill each `g()` span according to either the natural-language description or the provided regex pattern.

## Task

I will supply a **fully completed response**.
Your job is to generate the **corresponding query**, masking certain spans using the `g()` function.

## Requirements

* The generated query must hide **between 1 and 10 spans**.
* A span may be a word, phrase, sentence, paragraph, code block, symbol sequence, or any meaningful structural unit.
* Within each `g()` call, you may use:

  * `desc` only,
  * `regex` only,
  * both,
  * or neither (empty `g()`), depending on what best tests the model.
* The masked regions should be meaningful and non-trivial, targeting genuine infilling difficulty.
* **The final query must include at least two `g()` calls that specify a `regex=` pattern.**
* You must ensure the generated query is a valid Python f-string.
* You must wrap the entire query like f"""xxx"""
* Do not contain any explanations or additional text outside the f-string.
'''

DEMOS: list[dict[str, str]] = [
    {"role": "user", "content": "The capital of France is Paris."},
    {
        "role": "assistant",
        "content": 'f"""The capital of {g(desc="a country")} is {g(desc="a city", regex="[A-Z][a-z]+")}."""',
    },
    {"role": "user", "content": "E = mc^2 is Einstein's famous equation relating energy and mass."},
    {
        "role": "assistant",
        "content": 'f"""{g(desc="a physics equation", regex="[A-Z] = [a-z]+\\^[0-9]")} is Einstein\'s famous equation relating {g(desc="a physical quantity")} and {g(desc="another physical quantity")}."""',
    },
    {"role": "user", "content": "To bake a cake, you need flour, sugar, eggs, and butter."},
    {
        "role": "assistant",
        "content": 'f"""To bake a {g(desc="a type of dessert")}, you need {g(desc="an ingredient")}, {g(desc="another ingredient")}, {g(desc="another ingredient")}, and {g(desc="another ingredient")}."""',
    },
]

SEED = 42
BASE_URL = "https://api.siliconflow.cn/v1"
TIMEOUT = 20.0
MODELS = [
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "zai-org/GLM-4.6",
    "moonshotai/Kimi-K2-Instruct-0905",
    "Pro/deepseek-ai/DeepSeek-V3.2-Exp",
    "Kwaipilot/KAT-Dev",
    "inclusionAI/Ling-1T",
    "MiniMaxAI/MiniMax-M2",
]
MAX_TOKENS = 2048
TEMPERATURE = 0.7
PROCESSES = 10
COUNT = 2500
RANDOM_ARTICLES_DATASET = "Sculpt-AI/random-articles"
UPLOAD_DATASET_NAME = "Sculpt-AI/GIMBench"
SUBSET_NAME = "regex"
SPLIT_NAME = "test"


client = OpenAI(base_url=BASE_URL, timeout=TIMEOUT)
articles = load_dataset(RANDOM_ARTICLES_DATASET, split="train")


def init_worker():
    global client
    # Each worker has its own client instance
    client = OpenAI(base_url=BASE_URL, timeout=TIMEOUT)
    # Ensure different random seeds for each worker
    random.seed(SEED + current_process().pid)


def model_request(client: OpenAI, model: str, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        *DEMOS,
        {"role": "user", "content": prompt},
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            extra_body={"prefix": 'f"""'},
        )
    except Exception as e:
        print(f"Error during model request: {e}")
        return ""

    response_content = response.choices[0].message.content
    if not response_content:
        return ""
    return 'f"""' + response_content


def parse_query_response(query: str, response: str) -> tuple[Query, Response]:
    gim_query = Query(query)

    gim_response_regex = "".join(
        "(" + re.escape(part) + ")" if isinstance(part, str) else r"([\s\S]*?)"
        for part in gim_query.parts[1:-1]  # Exclude the <GIM_QUERY> and </GIM_QUERY>
    )
    matches = re.fullmatch(gim_response_regex, response)
    if not matches:
        raise ValueError("Response does not match the query structure.")
    gim_response_parts: list[str | MaskedTag] = []
    for i, part in enumerate(gim_query._parts[1:-1]):
        if isinstance(part, str):
            gim_response_parts.append(part)
        elif isinstance(part, MaskedTag):
            tag = deepcopy(part)
            tag.content = matches[i + 1]
            gim_response_parts.append(tag)
    gim_response = Response(gim_response_parts)
    return gim_query, gim_response


def generate_gim_query(_=None) -> dict:
    model = random.choice(MODELS)
    row = random.choice(articles)
    gim_response = row["article"]
    gim_query = model_request(client, model, gim_response)
    try:
        gim_query_obj, gim_response_obj = parse_query_response(eval(gim_query), gim_response)
    except Exception as e:
        print(f"Error parsing query/response: {e}")
        return {
            "gim_query": gim_query,
            "gim_response": gim_response,
            "model": model,
            "correct": False,
            "type": row["type"],
            "keywords": row["keywords"],
            "language": row["language"],
        }
    return {
        "gim_query": str(gim_query_obj),
        "gim_response": str(gim_response_obj),
        "model": model,
        "correct": True,
        "type": row["type"],
        "keywords": row["keywords"],
        "language": row["language"],
    }


if __name__ == "__main__":
    ds = []
    with Pool(processes=PROCESSES, initializer=init_worker) as pool:
        for result in tqdm(pool.imap_unordered(generate_gim_query, range(COUNT)), total=COUNT):
            if result:
                ds.append(result)
    hf_ds = Dataset.from_list(ds).filter(lambda x: x["correct"] is True).remove_columns(["correct"])
    hf_ds.save_to_disk(UPLOAD_DATASET_NAME.replace("/", "_") + f"_{SUBSET_NAME}_{SPLIT_NAME}")
    hf_ds.push_to_hub(UPLOAD_DATASET_NAME, SUBSET_NAME, split=SPLIT_NAME)
    print(
        f"Pushed {len(hf_ds)} records to dataset {UPLOAD_DATASET_NAME} subset {SUBSET_NAME} split {SPLIT_NAME}."
    )
