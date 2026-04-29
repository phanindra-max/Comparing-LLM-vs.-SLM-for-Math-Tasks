# Data

This directory stores local copies of the Kaggle competition data and derived data files used by the notebooks.

Expected files include:

- `train.csv`
- `test.csv`
- `sample_submission.csv`
- `train_augmented.csv`
- `train_pp.csv`
- `test_pp.csv`

The asset script writes downloaded data here:

```bash
./scripts/get_assets.sh <MODELS_FOLDER_ID> <DATA_FOLDER_ID>
```

If you publish a release, record the source, checksum, and license/competition terms for each data file.
