# Contributing

Thanks for your interest in improving this project.

## Development Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Install optional LLM dependencies only when working on Llama, Bedrock, or vLLM-style workflows:

```bash
pip install -r requirements-llm.txt
```

## Repository Conventions

- Keep reusable code in `src/llmvsslm/`.
- Keep notebooks in the matching `notebooks/` subfolder.
- Keep original class-project material under `archive/individual-projects/`.
- Do not commit large model artifacts, credentials, `.env` files, or generated caches.
- Record new experiment results in `EXPERIMENTS.md` with source path, data variant, metric, and caveats.

## Reporting Results

When adding or updating a model result, include:

- Model name and model family.
- Dataset variant and preprocessing steps.
- Train/validation/test split details.
- Public leaderboard score if applicable.
- Internal validation metrics.
- Hardware, Python version, package versions, and random seed.
- Link to the notebook or script that produced the result.

## Pull Requests

Before opening a pull request:

1. Run any affected notebooks or scripts when practical.
2. Confirm paths are repository-relative or controlled by documented environment variables.
3. Update `README.md`, `EXPERIMENTS.md`, or `REPRODUCIBILITY.md` when behavior or results change.
4. Avoid committing generated model files under `outputs/models/`.
