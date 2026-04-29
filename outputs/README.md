# Outputs

This directory stores generated outputs and local model artifacts.

Tracked CSV files are example Kaggle-style submissions from the class project. Large model artifacts should live under `outputs/models/` locally and should not be committed to Git.

Expected model layout for the demo:

```text
outputs/models/
├── mathBERT/
└── ensemble/
    ├── llama_1b_model/
    ├── deberta-model/
    └── t5-model/
```

For reproducible releases, archive model artifacts externally and document checksums in `REPRODUCIBILITY.md` or a release manifest.
