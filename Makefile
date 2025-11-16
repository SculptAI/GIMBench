install:
	uv sync

lint:
	uv run ruff check
	uv run ruff format --diff
	uv run mypy .

lint-fix:
	uv run ruff check --fix
	uv run ruff format
pre-commit:
	uv run pre-commit run --all-files
