# GIMBench

GIMBench is a benchmarking framework for evaluating Guided Infilling Models (GIM).

## Overview

This project provides tools and benchmarks to evaluate models' ability to perform guided infilling tasks - generating text that follows specific constraints and patterns.

## Installation

Install GIMBench using `uv`:

```bash
make install
```

For development, install with dev dependencies:

```bash
make install-dev
```

## Usage

GIMBench provides several benchmark types:

- **CV Parsing**: Evaluate models on structured information extraction from CVs
- **Regex Matching**: Test models' ability to generate text matching specific patterns
- **Multiple Choice QA**: Assess guided generation in question-answering contexts
- **Perplexity**: Measure language modeling quality with constraints

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Copyright

Copyright (c) 2025 SculptAI
