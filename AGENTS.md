# Repository Guidelines

## Project Structure & Module Organization
- Root: training code is vendored in `fairseq/` (task defs, CLI, preprocess); analysis scripts live in `analysis/` (evaluation, rationalization); small demo inputs in `sample-data/`.
- Scripts: `career_v1.sh` provides an end-to-end run pipeline; read it for environment and Slurm examples.
- Dependencies: Python 3.9+, GPU recommended. Install via `requirements.txt` and editable install of `fairseq/`.

## Build, Test, and Development Commands
- Setup environment:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
  - `cd fairseq && pip install --editable ./ && cd ..`
- Quick sanity checks:
  - `python analysis/compute_survey_data_perplexity.py --help`
  - Small run on toy data: set `BINARY_DATA_DIR=./sample-data`, then run an analysis script with `--model-name bag-of-jobs` to validate wiring.
- Training (example): run within `fairseq/` using `fairseq-train ...` as in `README.md` (use small `--max-update` for smoke tests).

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indentation, limit lines to ~100 chars, prefer type hints and argparse for CLIs.
- Naming: modules/files `snake_case.py`; constants `UPPER_SNAKE`; functions/vars `lower_snake`.
- I/O: keep paths configurable via flags or env vars (e.g., `BINARY_DATA_DIR`, `SAVE_DIR`, `LOG_DIR`). Do not hard-code absolute paths.

## Testing Guidelines
- Use `sample-data/` for deterministic, fast smoke tests; set `--max-update` small and fix `--seed`.
- When modifying `fairseq/` components, add or update tests under `fairseq/tests/` and run with `pytest -q` (if configured in your env).
- Include a short script/command in PRs to reproduce key metrics (e.g., validation loss or perplexity on a tiny split).

## Commit & Pull Request Guidelines
- Commits: imperative, concise subjects (<= 72 chars). Suggested scopes: `analysis:`, `fairseq:`, `preprocess:`, `scripts:`.
  - Example: `analysis: improve survey perplexity reporting`
- PRs: include description, linked issues, exact commands used, dataset identifier, seed, and before/after metrics (e.g., validation perplexity). Add logs or screenshots (TensorBoard) when relevant.

## Security & Configuration Tips
- Do not commit data, checkpoints, or large artifacts. Keep them under scratch paths; `.gitignore` them when needed.
- GPU strongly recommended; if training fails on FP16, try lowering LR or disable `--fp16` per README notes.
