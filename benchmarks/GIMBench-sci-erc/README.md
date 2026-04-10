# GIMBench SciERC Dataset

This folder contains the code used to convert the official SciERC raw release into a Hugging Face dataset that can be pushed to `Sculpt-AI/GIMBench-sci-erc`.

Source:
- https://nlp.cs.washington.edu/sciIE/

Usage:

```bash
python benchmarks/GIMBench-sci-erc/1_build_dataset.py \
    --raw_dir benchmarks/GIMBench-sci-erc/data/raw_data \
    --repo_id Sculpt-AI/GIMBench-sci-erc \
    --push_to_hub
```
