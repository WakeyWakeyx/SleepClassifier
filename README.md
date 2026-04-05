# Sleep Stage Classification with DREAMT E4 Signals

This project is a clean PyTorch baseline for sleep stage classification using the DREAMT 64 Hz aligned wearable dataset. It trains a window-based 1D CNN on the eight E4 channels:

- `BVP`
- `ACC_X`
- `ACC_Y`
- `ACC_Z`
- `TEMP`
- `EDA`
- `HR`
- `IBI`

The supervised target is `Sleep_Stage`, with default training on:

- `W`
- `N1`
- `N2`
- `N3`
- `R`

Rows and windows dominated by `P` or `Missing` are excluded from supervised training by default.

## Project Structure

```text
SleepClassifier/
├── dataset/
│   └── data_64Hz/
├── outputs/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loading.py
│   │   ├── preprocessing.py
│   │   └── windowing.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── model.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   └── train.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── checkpointing.py
│   │   ├── metrics.py
│   │   └── utilities.py
│   ├── __init__.py
│   ├── config.py
│   └── main.py
├── .dockerignore
├── .gitignore
├── compose.yaml
├── Dockerfile
├── README.md
└── requirements.txt
```

`src/latest.pt` and `src/best.pt` are generated automatically during training.

## Dataset Expectations

Each CSV is treated as one participant/session file. The loader recursively scans the configured dataset directory and validates that every file contains:

- `TIMESTAMP`
- `BVP`
- `ACC_X`
- `ACC_Y`
- `ACC_Z`
- `TEMP`
- `EDA`
- `HR`
- `IBI`
- `Sleep_Stage`

Other columns such as apnea annotations may exist, but they are ignored by the baseline pipeline.

By default, local training uses:

```text
D:\Dreamt\data_64Hz
```

You can override this with the environment variable:

```powershell
$env:DREAMT_DATASET_DIR = "D:\Dreamt\data_64Hz"
```

For Docker, the container uses `/app/dataset/data_64Hz`, and `compose.yaml` mounts your host dataset there.

## Preprocessing Behavior

The pipeline is defensive and keeps temporal order intact:

1. `TIMESTAMP` is coerced to numeric and used for sorting only.
2. Required columns are validated before training begins.
3. Feature columns are converted to numeric with coercion.
4. `Sleep_Stage` values are cleaned and validated against the known label set.
5. Missing feature values are filled per participant.

Implemented fill strategy:

- `IBI`: forward fill, then backward fill, then `0.0` if still missing.
- Other aligned channels: linear interpolation, then forward fill, then backward fill, then `0.0` fallback.

This works with 64 Hz aligned DREAMT files even though the original channels came from different native rates.

## Windowing and Labels

Training uses fixed windows over each participant file. Default settings are in [`src/config.py`](./src/config.py):

- `window_length = 256`
- `step = 128`

Each sample is shaped as `[channels, time]`, with channels ordered as:

1. `BVP`
2. `ACC_X`
3. `ACC_Y`
4. `ACC_Z`
5. `TEMP`
6. `EDA`
7. `HR`
8. `IBI`

Window labels are assigned by majority vote over `Sleep_Stage` inside the window.

Conservative default label policy:

- `P` and `Missing` are excluded from the supervised label set.
- At least 80% of the window must contain valid supervised labels.
- Among those valid labels, at least 80% must agree on the winning class.
- Windows that fail those rules are dropped.

The dataset also checks timestamp continuity and drops windows that span clear temporal gaps.

## Train / Validation / Test Splitting

Splitting is done by participant file, never by pooled rows or pooled windows. The default ratio is:

- 70% train files
- 15% validation files
- 15% test files

The split is deterministic and saved to:

```text
outputs/split_manifest.json
```

If the set of discovered files stays the same, the saved manifest is reused so repeated runs stay aligned.

## Normalization

Normalization statistics are computed from train participants only, then reused for validation and test without leakage.

Saved artifact:

```text
outputs/normalization_stats.json
```

## Model

The baseline model is a multichannel 1D CNN implemented in [`src/models/model.py`](./src/models/model.py). It uses:

- stacked `Conv1d + BatchNorm + ReLU` blocks
- temporal pooling
- dropout
- a compact classifier head

The output layer predicts exactly five classes:

- `W`
- `N1`
- `N2`
- `N3`
- `R`

## Training Behavior

Training is launched with:

```powershell
python -m src.main
```

The training loop includes:

- class weights computed from train windows only
- `AdamW`
- weighted cross-entropy loss
- early stopping on validation macro F1
- latest and best checkpoints
- validation and test metrics
- confusion matrix plots

Main artifacts:

- `src/latest.pt`
- `src/best.pt`
- `outputs/train.log`
- `outputs/training_history.json`
- `outputs/class_weights.json`
- `outputs/dataset_summary.json`
- `outputs/validation/validation_latest_metrics.json`
- `outputs/validation/validation_latest_confusion_matrix.png`
- `outputs/validation/validation_best_metrics.json`
- `outputs/validation/validation_best_confusion_matrix.png`
- `outputs/test/test_metrics.json`
- `outputs/test/test_confusion_matrix.png`
- `outputs/run_summary.json`

## Local Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Point to the dataset if needed

If your CSV files already live in `D:\Dreamt\data_64Hz`, the default config is fine.

If not, set:

```powershell
$env:DREAMT_DATASET_DIR = "C:\path\to\data_64Hz"
```

### 4. Run training

```powershell
python -m src.main
```

## Docker Usage

The Docker image is CUDA-friendly and intended for GPU training when the host has NVIDIA Container Toolkit configured.

### 1. Set the host dataset path

PowerShell example:

```powershell
$env:DREAMT_HOST_DATASET_DIR = "D:/Dreamt/data_64Hz"
```

Use forward slashes in the Docker bind path on Windows to avoid quoting issues.

### 2. Build and run

```powershell
docker compose up --build
```

The container will mount your host dataset into:

```text
/app/dataset/data_64Hz
```

and will run:

```text
python -m src.main
```

## Hyperparameter Changes

Edit [`src/config.py`](./src/config.py) to change:

- dataset path
- window length and step
- label filtering thresholds
- train/validation/test ratios
- CNN channels and kernel sizes
- batch size
- learning rate
- weight decay
- early stopping patience

## Troubleshooting

### Dataset directory not found

Set `DREAMT_DATASET_DIR` for local Python or `DREAMT_HOST_DATASET_DIR` for Docker.

### Zero windows after preprocessing

Common causes:

- windows are too short for the selected `window_length`
- `min_valid_fraction` is too strict
- `label_agreement_threshold` is too strict
- timestamp gaps are causing windows to be dropped

### A class is missing from the train split

The training code raises an error if one of `W/N1/N2/N3/R` has zero train windows, because class-weighted training would be ill-defined. In that case, adjust the random seed or review dataset coverage.

### GPU is not visible in Docker

Check:

- Docker Desktop is using Linux containers
- NVIDIA drivers are installed
- NVIDIA Container Toolkit is configured
- `docker compose` is allowed to request GPUs on your machine

### Running on CPU

The code automatically falls back to CPU if CUDA is unavailable.

## Notes

- `TIMESTAMP` is never used as a direct predictive feature in the baseline.
- Apnea event columns are intentionally excluded from the baseline model.
- The code is written to favor correctness, explicit validation, and reproducible participant-level experiments over overly clever shortcuts.
