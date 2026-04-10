import argparse
import shutil
import tarfile
import urllib.request

from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict


RAW_DATA_URL = "https://nlp.cs.washington.edu/sciIE/data/sciERC_raw.tar.gz"
DEFAULT_RAW_DIR = Path(__file__).parent / "data" / "raw_data"
DEFAULT_REPO_ID = "Sculpt-AI/GIMBench-sci-erc"
SPLIT_SIZES = {"train": 350, "dev": 50, "test": 100}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_ann(path: Path) -> tuple[list[dict], list[dict], list[list[str]]]:
    entities_by_id: dict[str, dict] = {}
    relations: list[dict] = []
    coref_edges: list[list[str]] = []

    for line in _read_text(path).splitlines():
        line = line.strip()
        if not line:
            continue

        tag, payload = line.split("\t", maxsplit=1)
        if tag.startswith("T"):
            label_span, entity_text = payload.split("\t", maxsplit=1)
            parts = label_span.split()
            if len(parts) < 3:
                continue
            label = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            entities_by_id[tag] = {
                "id": tag,
                "label": label,
                "start": start,
                "end": end,
                "text": entity_text,
            }
        elif tag.startswith("R"):
            relation_parts = payload.split()
            if len(relation_parts) != 3:
                continue
            relation_label, arg1, arg2 = relation_parts
            left_id = arg1.split(":", maxsplit=1)[1]
            right_id = arg2.split(":", maxsplit=1)[1]
            if relation_label == "COREF":
                coref_edges.append([left_id, right_id])
            else:
                relations.append(
                    {
                        "relation": relation_label,
                        "arg1": left_id,
                        "arg2": right_id,
                    }
                )

    return list(entities_by_id.values()), relations, coref_edges


def _build_coreference_clusters(edges: list[list[str]]) -> list[list[str]]:
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in edges:
        union(left, right)

    clusters: dict[str, list[str]] = defaultdict(list)
    for node in parent:
        clusters[find(node)].append(node)

    return [sorted(cluster) for cluster in clusters.values()]


def _normalize_sample(sample: dict) -> dict:
    text = _read_text(sample["txt_path"])
    entities, relations, coref_edges = _read_ann(sample["ann_path"])
    entity_index = {entity["id"]: entity for entity in entities}

    gold_relations = []
    for relation in relations:
        left_entity = entity_index.get(relation["arg1"])
        right_entity = entity_index.get(relation["arg2"])
        if not left_entity or not right_entity:
            continue
        gold_relations.append(
            {
                "head": left_entity["text"],
                "head_span": [left_entity["start"], left_entity["end"]],
                "tail": right_entity["text"],
                "tail_span": [right_entity["start"], right_entity["end"]],
                "relation": relation["relation"],
            }
        )

    coreference_clusters = [
        [entity_index[entity_id]["text"] for entity_id in cluster if entity_id in entity_index]
        for cluster in _build_coreference_clusters(coref_edges)
    ]

    tokens = text.split()

    return {
        "doc_key": sample["doc_key"],
        "sentences": [tokens],
        "tokens": tokens,
        "text": text,
        "entities": entities,
        "gold_relations": gold_relations,
        "coreference_clusters": coreference_clusters,
    }


def _resolve_raw_dir(raw_dir: Path) -> Path:
    if raw_dir.exists() and any(raw_dir.glob("*.ann")) and any(raw_dir.glob("*.txt")):
        return raw_dir

    candidate = raw_dir / "raw_data"
    if candidate.exists() and any(candidate.glob("*.ann")) and any(candidate.glob("*.txt")):
        return candidate

    alt_candidate = raw_dir.parent / "raw_data"
    if alt_candidate.exists() and any(alt_candidate.glob("*.ann")) and any(alt_candidate.glob("*.txt")):
        return alt_candidate

    return raw_dir


def _ensure_raw_data(raw_dir: Path, download: bool) -> Path:
    raw_dir = _resolve_raw_dir(raw_dir)
    if raw_dir.exists() and any(raw_dir.glob("*.ann")) and any(raw_dir.glob("*.txt")):
        return raw_dir

    if not download:
        raise FileNotFoundError(
            f"Missing SciERC raw files in {raw_dir}. Use --download or place the extracted raw_data directory there."
        )

    raw_dir.parent.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir.parent / "sciERC_raw.tar.gz"
    urllib.request.urlretrieve(RAW_DATA_URL, archive_path)

    extract_root = raw_dir.parent / "_scierc_extract"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_root)

    extracted_candidate = extract_root / "raw_data"
    if not extracted_candidate.exists():
        raise FileNotFoundError(f"Could not find raw_data directory after extracting {archive_path}")

    if raw_dir.exists():
        shutil.rmtree(raw_dir)

    extracted_candidate.rename(raw_dir)
    return raw_dir


def build_dataset(raw_dir: Path) -> DatasetDict:
    doc_files = sorted(raw_dir.glob("*.txt"))
    samples = [
        {
            "doc_key": txt_file.stem,
            "txt_path": txt_file,
            "ann_path": txt_file.with_suffix(".ann"),
        }
        for txt_file in doc_files
        if txt_file.with_suffix(".ann").exists()
    ]

    rows = sorted((_normalize_sample(sample) for sample in samples), key=lambda row: row["doc_key"])

    expected_total = sum(SPLIT_SIZES.values())
    if len(rows) < expected_total:
        raise ValueError(f"Expected at least {expected_total} SciERC docs, found {len(rows)}")

    train_end = SPLIT_SIZES["train"]
    dev_end = train_end + SPLIT_SIZES["dev"]
    test_end = dev_end + SPLIT_SIZES["test"]

    return DatasetDict(
        {
            "train": Dataset.from_list(rows[:train_end]),
            "dev": Dataset.from_list(rows[train_end:dev_end]),
            "test": Dataset.from_list(rows[dev_end:test_end]),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and upload the SciERC Hugging Face dataset.")
    parser.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory with raw SciERC files")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id")
    parser.add_argument("--download", action="store_true", help="Download the official SciERC archive if needed")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the generated DatasetDict to the Hub")
    parser.add_argument(
        "--save_to_disk",
        type=Path,
        default=None,
        help="Optional local directory to save the generated DatasetDict",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raw_dir = _ensure_raw_data(args.raw_dir, args.download)

    dataset_dict = build_dataset(raw_dir)
    for split_name, split_dataset in dataset_dict.items():
        print(f"Loaded {len(split_dataset)} records for split '{split_name}'.")
        if len(split_dataset) > 0:
            print(split_dataset[0])

    if args.save_to_disk is not None:
        args.save_to_disk.mkdir(parents=True, exist_ok=True)
        dataset_dict.save_to_disk(str(args.save_to_disk))
        print(f"Saved dataset to {args.save_to_disk}")

    if args.push_to_hub:
        dataset_dict.push_to_hub(args.repo_id)
        print(f"Pushed dataset to {args.repo_id}")
