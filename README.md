# GIMBench

GIMBench is a benchmarking framework for evaluating Guided Infilling Models (GIM).

## Overview

This project provides tools and benchmarks to evaluate models' ability to perform guided infilling tasks - generating text that follows specific constraints and patterns.

## Installation

Install GIMBench using pip:

```bash
pip install gimbench
```

For development:

```bash
make install-dev
```

## Usage

GIMBench provides several benchmark types:

- **CV Parsing**: Evaluate models on structured information extraction from CVs
- **Regex Matching**: Test models' ability to generate text matching specific patterns
- **Multiple Choice QA**: Assess guided generation in question-answering contexts
- **Perplexity**: Measure language modeling quality with constraints
- **Code Infilling**: Evaluate code infilling via unit-test execution (pass@k)
- **SciERC Relation Extraction**: Evaluate scientific relation extraction on the Hugging Face dataset `Sculpt-AI/GIMBench-sci-erc`

### Example Commands

Run MMLU-Pro benchmark:

```bash
python -m gimbench.mcqa.mmlu_pro \
    --model_type vllm \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --base_url http://localhost:8000/v1
```

Run GPQA Diamond benchmark:

```bash
python -m gimbench.mcqa.gpqa_diamond \
    --model_type openai \
    --model_name gpt-4 \
    --api_key YOUR_API_KEY
```

Run GIM-SFT perplexity evaluation:

```bash
python -m gimbench.ppl.gim_sft \
    --model_type vllm-offline \
    --model_name meta-llama/Llama-3.1-8B-Instruct
```

Run HumanEval Infilling benchmark (code generation + unit-test execution, pass@k):

```bash
# GIM-guided infilling (default), pass@1
python -m gimbench.code.humaneval_infilling \
    --model_type vllm \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --base_url http://localhost:8000/v1

# Sample 20 completions per problem, report pass@1 and pass@10
python -m gimbench.code.humaneval_infilling \
    --model_type vllm \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --base_url http://localhost:8000/v1 \
    --temperature 0.8 \
    --num_samples 20 \
    --pass_k 1 10

# Plain LLM (no GIMKit)
python -m gimbench.code.humaneval_infilling \
    --model_type vllm \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --base_url http://localhost:8000/v1 \
    --no_gimkit

Run SciERC relation extraction benchmark (Hugging Face dataset):

```bash
python -m gimbench.scierc.scierc \
    --model_type vllm \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --base_url http://localhost:8000/v1 \
    --scierc_split test

# Plain LLM (no GIMKit)
python -m gimbench.scierc.scierc \
    --model_type vllm \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --base_url http://localhost:8000/v1 \
    --scierc_split dev \
    --no_gimkit
```

If you need to rebuild and upload the dataset, use:

```bash
python benchmarks/GIMBench-sci-erc/1_build_dataset.py \
    --raw_dir benchmarks/GIMBench-sci-erc/data/raw_data \
    --repo_id Sculpt-AI/GIMBench-sci-erc \
    --push_to_hub
```
```

## Development

Run linting:

```bash
make lint
```

Fix linting issues automatically:

```bash
make lint-fix
```

Run pre-commit hooks:

```bash
make pre-commit
```
