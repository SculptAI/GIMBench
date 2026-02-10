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
