# Reproducibility

This repository contains both executable demo code and notebook-based experiments. The demo path is the most stable entry point; full model reproduction depends on GPU availability, downloaded artifacts, and optional LLM tooling.

## Environment

Use Python 3.10 or 3.11 for the broadest compatibility with PyTorch, Transformers, Streamlit, and the optional LLM packages.

Install the core dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Install optional LLM/data-generation dependencies only when needed:

```bash
pip install -r requirements-llm.txt
```

GPU-backed experiments should record:

- GPU model and memory.
- CUDA and driver versions.
- Python version.
- Package versions from `pip freeze`.
- Random seeds and training configuration.

## Assets

The asset download script expects two Google Drive folder IDs:

```bash
./scripts/get_assets.sh <MODELS_FOLDER_ID> <DATA_FOLDER_ID>
```

The currently documented class-project IDs are:

```bash
./scripts/get_assets.sh 1ancje2FsGw9dTMMCXCO2CfuXsqgDJKBE 1OpVGWl8JlRD3G3mB8IqoAE6DY1bd08sv
```

Expected local layout after download:

```text
data/
├── train.csv
├── test.csv
└── ...

outputs/
└── models/
    ├── mathBERT/
    └── ensemble/
        ├── llama_1b_model/
        ├── deberta-model/
        └── t5-model/
```

For publication-quality replication, add checksums or archive artifacts in a DOI-backed repository such as Zenodo.

## Run Order

1. Download or restore the Kaggle data under `data/`.
2. Optional: run data generation notebooks under `notebooks/01_data_generation/`.
3. Optional: run augmentation with `src/llmvsslm/augment.py`.
4. Train or evaluate individual model families from the notebooks listed in `EXPERIMENTS.md`.
5. Run the ensemble notebook after its component models are saved.
6. Launch the local demo with `scripts/run_demo.sh`.

## Demo

After installing dependencies and downloading model artifacts:

```bash
./scripts/run_demo.sh
```

The demo serves `src/llmvsslm/app.py` on `http://localhost:8888` and loads models from `outputs/models/` by default.

You can override model and data locations with environment variables:

```bash
export LMVSSLM_MODEL_DIR=/path/to/models
export LMVSSLM_DATA_DIR=/path/to/data
```

## Known Sources Of Non-Determinism

- AWS Bedrock data generation depends on provider-side model behavior and credentials.
- vLLM/Gemma paraphrase generation can vary by model version, sampling settings, and hardware.
- Some notebooks were originally executed in Kaggle-style environments and may need path edits before rerun.
- The root requirements are intentionally usable for the demo but are not a full lockfile.
- Kaggle public leaderboard scores depend on the competition test split and submission process.
