import json
import re
import shutil

from pathlib import Path

from datasets import Dataset
from schemas import get_hf_features


def clean_filename(name):
    # Keep only alphanumeric, space, hyphen, underscore
    clean = re.sub(r"[^\w\s-]", "", name)
    # Replace spaces with underscores
    clean = re.sub(r"\s+", "_", clean)
    return clean.strip()


def create_dataset():
    base_dir = Path(__file__).parent
    input_file = base_dir / "data" / "parsed_fields.jsonl"
    pdf_storage = base_dir / "data" / "hf_dataset_pdfs"
    pdf_storage.mkdir(parents=True, exist_ok=True)

    with open(input_file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    dataset_rows = []

    print(f"Preparing {len(records)} records for dataset...")

    for record in records:
        if "parsed_data" not in record:
            continue

        original_path = Path(record["original_path"])
        if not original_path.exists():
            print(f"Warning: Original file missing {original_path}")
            continue

        parsed = record["parsed_data"]
        name = parsed.get("name")

        if not name or name.strip() == "":
            new_filename = original_path.name  # Fallback
        else:
            new_filename = clean_filename(name) + original_path.suffix

        target_path = pdf_storage / new_filename

        # Resolve collision
        counter = 1
        stem = target_path.stem
        while target_path.exists():
            target_path = pdf_storage / f"{stem}_{counter}{target_path.suffix}"
            counter += 1

        shutil.copy2(original_path, target_path)

        # Flattened row (schema is already flat)
        row = {
            "pdf": str(target_path),
            "file_name": target_path.name,
            "extracted_text": record.get("extracted_text", ""),
            **parsed,
        }
        dataset_rows.append(row)

    # Get Features from schema definitions
    features = get_hf_features()

    # Create HF Dataset
    # We can load from dicts
    ds = Dataset.from_list(dataset_rows, features=features)
    ds.push_to_hub("Sculpt-AI/GIMBench-cv-parse")


if __name__ == "__main__":
    create_dataset()
