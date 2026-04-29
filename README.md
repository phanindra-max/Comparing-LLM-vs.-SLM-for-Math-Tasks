# LLM vs SLM for Math Problem Classification

This repository compares large language model style workflows, smaller transformer classifiers, and classical machine learning baselines for classifying mathematical word problems into eight subject areas. The work was originally developed for an NLP class project using the Kaggle competition [Classification of Math Problems by Kasut Academy](https://www.kaggle.com/competitions/classification-of-math-problems-by-kasut-academy/overview).

The strongest documented result is a hard-voting ensemble over Llama 3.2 1B, T5-base, and DeBERTa-v3-base with a public leaderboard score of `0.8588`.

## Key Results

| Model or workflow | Type | Data variant | Public score | Source |
| --- | --- | --- | --- | --- |
| Llama 3.2 1B + T5-base + DeBERTa-v3-base ensemble | Ensemble | Mixed saved model outputs | `0.8588` | [`notebooks/05_ensemble/nlp-final-project-ensemble-model.ipynb`](notebooks/05_ensemble/nlp-final-project-ensemble-model.ipynb) |
| DeBERTa-v3-base augmented | Smaller transformer | Augmented training data | `0.8510` | [`notebooks/03_slm_transformers/nlp-final-project-deberta-v3-base-augmented.ipynb`](notebooks/03_slm_transformers/nlp-final-project-deberta-v3-base-augmented.ipynb) |
| DeBERTa-v3-base paraphrased | Smaller transformer | Paraphrased training data | `0.8394` | [`notebooks/03_slm_transformers/nlp-final-project-nn-deberta-v3-paraphrased.ipynb`](notebooks/03_slm_transformers/nlp-final-project-nn-deberta-v3-paraphrased.ipynb) |
| Llama 3.2 1B + LoRA | LLM-style classifier | Instruction fine-tuning | `0.8346` | [`notebooks/04_llm_experiments/LLAMA_1B.ipynb`](notebooks/04_llm_experiments/LLAMA_1B.ipynb) |
| DeBERTa-v3-base | Smaller transformer | Original training data | `0.8326` | [`notebooks/03_slm_transformers/nlp-final-project-nn-deberta-v3.ipynb`](notebooks/03_slm_transformers/nlp-final-project-nn-deberta-v3.ipynb) |
| T5-base | Smaller seq2seq model | Original training data | `0.8239` | [`notebooks/03_slm_transformers/nlp-final-project-t5.ipynb`](notebooks/03_slm_transformers/nlp-final-project-t5.ipynb) |
| MathBERT | Domain transformer baseline | Original training data | `0.8152` | [`archive/individual-projects/phanindra-kalaga-individual-project/Individual-Final-Project-Report/phanindra-kalaga-final-project.md`](archive/individual-projects/phanindra-kalaga-individual-project/Individual-Final-Project-Report/phanindra-kalaga-final-project.md) |
| Classical ML baseline | Classical ML | TF-IDF style features | `0.7862` | [`notebooks/02_classical_baselines/nlp-final-project-classical.ipynb`](notebooks/02_classical_baselines/nlp-final-project-classical.ipynb) |

Scores are copied from notebook/report markdown and should be treated as the documented public leaderboard record for this class project. See [`EXPERIMENTS.md`](EXPERIMENTS.md) for caveats and replication notes.

## Repository Map

- [`src/llmvsslm/`](src/llmvsslm/) contains the Streamlit demo, model loading utilities, MathBERT training script, augmentation script, and baseline scripts.
- [`notebooks/`](notebooks/) contains the research notebooks grouped by data generation, classical baselines, smaller transformer experiments, LLM experiments, and ensemble experiments.
- [`data/`](data/) contains the Kaggle CSV files and derived training data when downloaded or restored locally.
- [`outputs/`](outputs/) contains example submission files and downloaded model artifacts under `outputs/models/`.
- [`scripts/`](scripts/) contains asset download and demo launch scripts.
- [`archive/`](archive/) preserves the original class-project submissions, duplicate notebooks, proposal, final report, and presentation materials for provenance.

## Quick Start

Create an environment and install the core demo dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Download data and saved model artifacts from Google Drive:

```bash
chmod +x scripts/get_assets.sh
./scripts/get_assets.sh 1ancje2FsGw9dTMMCXCO2CfuXsqgDJKBE 1OpVGWl8JlRD3G3mB8IqoAE6DY1bd08sv
```

The script writes models to `outputs/models/` and data to `data/`.

Run the local Streamlit demo:

```bash
chmod +x scripts/run_demo.sh
./scripts/run_demo.sh
```

The demo starts on `http://localhost:8888`.

## Reproducing Experiments

Start with [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for environment setup, asset expectations, and known sources of non-determinism. Use [`EXPERIMENTS.md`](EXPERIMENTS.md) as the index of model families, notebook paths, scores, and caveats.

Some experiments require optional heavy dependencies such as `unsloth`, `xformers`, `trl`, `boto3`, or vLLM-compatible GPU environments. Install those from [`requirements-llm.txt`](requirements-llm.txt) only when reproducing Llama, Bedrock, or vLLM/Gemma workflows.

## Citation

If this repository helps your work, cite it using [`CITATION.cff`](CITATION.cff).

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).
