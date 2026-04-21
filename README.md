# DREAMT Sleep Classifier Baseline

Production-style PyTorch baseline for ternary sleep-stage classification on the DREAMT wearable dataset using only the `data_64Hz` folder.

This project intentionally excludes:

- `data_100Hz`
- PSG-only signals
- apnea annotation columns as model inputs

## Scope

Target column:

- `Sleep_Stage`

Ternary label mapping:

- `W` -> `Awake` -> `0`
- `N1`, `N2` -> `Light Sleep` -> `1`
- `N3`, `R` -> `Deep Sleep` -> `2`

Excluded rows:

- `P`
- `Missing`
- missing `Sleep_Stage`

Input features used by the model:

- `BVP`
- `IBI`
- `EDA`
- `TEMP`
- `ACC_X`
- `ACC_Y`
- `ACC_Z`
- `HR`

Leakage-protected exclusions:

- `Sleep_Stage`
- `Obstructive_Apnea`
- `Central_Apnea`
- `Hypopnea`
- `Mixed_Apnea`

## Project Layout

The original requested layout is preserved with one small adjustment: reusable library code lives in the `sleep_classifier/` package, while the runnable scripts stay at the repository root.

```text
SleepClassifier/
  README.md
  requirements.txt
  train.py
  evaluate.py
  infer.py
  run_small_debug.py
  sleep_classifier/
    __init__.py
    config.py
    utils.py
    label_mapping.py
    data/
      __init__.py
      scan_dataset.py
      dataset.py
      preprocessing.py
      splits.py
    models/
      __init__.py
      cnn1d.py
```

## Dataset Expectations

Point `--dataset-root` at the DREAMT `data_64Hz` directory.

Each usable CSV must contain at least:

- `TIMESTAMP`
- `BVP`
- `IBI`
- `EDA`
- `TEMP`
- `ACC_X`
- `ACC_Y`
- `ACC_Z`
- `HR`
- `Sleep_Stage`

The loader:

- scans CSVs deterministically
- extracts participant ids from filenames
- skips bad files with warnings
- coerces numeric columns with `errors="coerce"`
- removes invalid labels and invalid timestamps
- drops rows where all wearable features are missing
- sorts by `TIMESTAMP`
- drops duplicate timestamps
- forward-fills then backward-fills wearable features within participant
- drops rows that still contain missing features afterward

## Windowing

Baseline windowing is aligned to 30-second sleep staging epochs:

- sample rate: `64 Hz`
- window length: `30 seconds`
- default window size: `1920 rows`
- default stride: `30 seconds`
- label rule: majority vote, with ties resolved by the last sample in the window
- windows never cross participant boundaries

By default, a window also needs at least `90%` of rows to have been fully observed before fill-based imputation.

## Splits And Normalization

- splitting is by participant only, never by row or window
- train/val/test defaults: `70/15/15`
- a deterministic seed is used
- split assignments are saved to JSON
- normalization stats are computed from training windows only
- class weights are computed from the training split only

## Install

Use Python 3.11+.

```bash
pip install -r requirements.txt
```

For RTX 4090 usage on Windows, make sure your installed PyTorch build includes CUDA support appropriate for your system.

## Train

```bash
python train.py --dataset-root data_64Hz --output-dir artifacts\baseline_run
```

Helpful overrides:

```bash
python train.py ^
  --dataset-root data_64Hz ^
  --output-dir artifacts\baseline_run ^
  --batch-size 64 ^
  --epochs 25 ^
  --num-workers 0 ^
  --stride-seconds 30
```

Features:

- automatic CUDA detection
- device logging at startup
- mixed precision with `torch.cuda.amp` when CUDA is available
- AdamW optimizer
- weighted cross entropy
- gradient clipping
- early stopping on validation macro F1
- best and last checkpoint saving
- resume support with `--resume-from`

## Debug Run

This is the safe end-to-end sanity path and is intentionally capped to only the first 10 CSV files after deterministic sorting.

```bash
python run_small_debug.py --dataset-root data_64Hz
```

This mode is suitable for pipeline verification only, not for reporting final metrics.

## Evaluate

Evaluate only on the held-out test split saved with training:

```bash
python evaluate.py ^
  --checkpoint artifacts\baseline_run\checkpoints\best_model.pt
```

Saved outputs include:

- JSON and text classification reports
- confusion matrix plot
- aggregate metrics including accuracy, balanced accuracy, macro precision, macro recall, and macro F1

## Inference

Run window-level inference for one participant CSV:

```bash
python infer.py ^
  --checkpoint artifacts\baseline_run\checkpoints\best_model.pt ^
  --input-csv data_64Hz\SID001_whole_df.csv
```

Optional CSV export:

```bash
python infer.py ^
  --checkpoint artifacts\baseline_run\checkpoints\best_model.pt ^
  --input-csv data_64Hz\SID001_whole_df.csv ^
  --output-csv artifacts\baseline_run\reports\SID001_predictions.csv
```

Prediction exports contain:

- `participant_id`
- `window_start_timestamp`
- `window_end_timestamp`
- `predicted_class_id`
- `predicted_class_name`
- `confidence`

## Reproducibility Notes

The code seeds:

- Python
- NumPy
- PyTorch

CUDA runs can still retain some nondeterminism depending on the exact kernels selected by cuDNN and mixed precision behavior. This project enables `cudnn.benchmark` and TF32 for performance on modern NVIDIA GPUs.
