# Sleep Stage Classification with DREAMT E4 Signals

This project is a PyTorch sleep stage classification baseline for the DREAMT 64 Hz aligned wearable dataset. It keeps the original `python -m src.main` execution path and now adds stronger diagnostics, configurable imbalance handling, better regularization controls, and a second baseline model option for temporal modeling.

The supervised target remains `Sleep_Stage` with five classes:

- `W`
- `N1`
- `N2`
- `N3`
- `R`

Excluded labels such as `P` and `Missing` are never used as supervised targets.

## Project Structure

```text
SleepClassifier/
|-- dataset/
|   `-- data_64Hz/
|-- outputs/
|-- src/
|   |-- data/
|   |   |-- data_loading.py
|   |   |-- preprocessing.py
|   |   `-- windowing.py
|   |-- models/
|   |   `-- model.py
|   |-- training/
|   |   |-- evaluate.py
|   |   `-- train.py
|   |-- utils/
|   |   |-- checkpointing.py
|   |   |-- metrics.py
|   |   `-- utilities.py
|   |-- config.py
|   `-- main.py
|-- Dockerfile
|-- compose.yaml
|-- requirements.txt
`-- README.md
```

`src/latest.pt` and `src/best.pt` are generated during training.

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

By default the project looks for data in:

```text
dataset/data_64Hz
```

Override that with:

```powershell
$env:DREAMT_DATASET_DIR = "C:\path\to\data_64Hz"
```

Inside Docker, the same project-relative path resolves to `/app/dataset/data_64Hz`.

## Preprocessing

The pipeline keeps participant files separate and applies preprocessing per file:

1. `TIMESTAMP` is coerced to numeric and used for sorting only.
2. Required columns are validated before training begins.
3. Features are converted to numeric with coercion.
4. `Sleep_Stage` is cleaned and validated against the configured label set.
5. Missing feature values are imputed per participant.

Fill strategy:

- `IBI`: forward fill, backward fill, then `0.0`
- all other aligned channels: linear interpolation, forward fill, backward fill, then `0.0`

Normalization statistics are computed from TRAIN participants only and saved to:

```text
outputs/normalization_stats.json
```

## Windowing and Conservative Labels

Window generation is configurable in [`src/config.py`](src/config.py).

Important settings:

- `window_length`
- `step`
- `label_purity_threshold`
- `min_valid_fraction`
- `drop_ambiguous_windows`
- `require_min_valid_labels`
- `continuity_gap_factor`

Default behavior is still conservative:

- `P` and `Missing` are excluded from supervised targets
- windows can be dropped if they contain too few valid supervised labels
- labels are assigned by majority vote over valid labels
- windows can be dropped if the majority label purity is below the configured threshold
- windows spanning large timestamp discontinuities are dropped

Window diagnostics are saved after generation for train, validation, and test, including:

- per-class retained window counts
- raw candidate windows per file
- kept windows per file
- discarded windows per file
- discard reasons such as temporal gaps, excluded-label contamination, and insufficient purity

Main artifact:

```text
outputs/dataset_summary.json
```

## Split Integrity and Leakage Guards

Splitting is done by participant file only, never by pooled windows. The pipeline explicitly checks that no file appears in more than one split.

The split manifest is saved to:

```text
outputs/split_manifest.json
```

That manifest includes:

- train/validation/test file lists
- file counts by split
- window counts by split
- window class counts by split
- integrity checks confirming split disjointness

Train-time statistics are restricted to TRAIN data only:

- normalization stats: TRAIN participants only
- class counts and class weights: TRAIN windows only
- weighted sampling: TRAIN windows only

## Model Options

Model selection is controlled through `config.model.model_type`.

Supported values:

- `cnn_baseline`
- `cnn_bilstm`

`cnn_baseline` uses a stronger residual-style 1D CNN with dropout and pooled temporal features.

`cnn_bilstm` uses the same CNN feature extractor and then runs a bidirectional LSTM over the reduced temporal sequence before classification.

The input shape remains:

```text
[batch, channels, time]
```

## Imbalance Handling and Loss Options

Training now supports configurable class-imbalance handling from [`src/config.py`](src/config.py):

- `use_weighted_sampler`
- `loss_name`
- `label_smoothing`
- `focal_gamma`
- `focal_use_class_weights`
- `focal_alpha`

Supported losses:

- `cross_entropy`
- `weighted_cross_entropy`
- `focal_loss`

Class weights are computed from TRAIN windows only and saved to:

```text
outputs/class_weights.json
```

## Training Behavior

Training still runs with:

```powershell
python -m src.main
```

The training loop includes:

- `AdamW`
- configurable dropout and weight decay
- optional label smoothing for cross-entropy losses
- optional `WeightedRandomSampler` for the training loader
- early stopping on validation macro F1
- best checkpoint selection by validation macro F1
- final test evaluation using the best checkpoint

## Evaluation Diagnostics

Validation and test evaluation now save richer diagnostics under `outputs/`.

Saved metrics include:

- loss
- accuracy
- balanced accuracy
- macro precision, recall, F1
- weighted precision, recall, F1
- per-class precision, recall, F1
- per-class support
- prediction distribution per class
- confusion matrices as JSON/CSV/PNG

Key artifacts:

- `outputs/training_history.json`
- `outputs/validation/validation_latest_metrics.json`
- `outputs/validation/validation_latest_summary.csv`
- `outputs/validation/validation_latest_per_class_metrics.csv`
- `outputs/validation/validation_latest_confusion_matrix.csv`
- `outputs/validation/validation_latest_confusion_matrix.png`
- `outputs/validation/validation_best_metrics.json`
- `outputs/validation/validation_best_confusion_matrix.png`
- `outputs/validation/epochs/validation_epoch_XXX_metrics.json`
- `outputs/validation/epochs/validation_epoch_XXX_confusion_matrix.png`
- `outputs/test/test_metrics.json`
- `outputs/test/test_summary.csv`
- `outputs/test/test_per_class_metrics.csv`
- `outputs/test/test_confusion_matrix.csv`
- `outputs/test/test_confusion_matrix.png`
- `outputs/run_summary.json`

These artifacts are intended to make class-collapse failure modes easy to inspect, especially for `N3` and `R`.

## Local Setup

1. Create a virtual environment.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

3. Set `DREAMT_DATASET_DIR` if your dataset is not under `dataset/data_64Hz`.

```powershell
$env:DREAMT_DATASET_DIR = "C:\path\to\data_64Hz"
```

4. Run training.

```powershell
python -m src.main
```

## Docker Usage

PowerShell example:

```powershell
$env:DREAMT_HOST_DATASET_DIR = "C:/path/to/data_64Hz"
docker compose up --build
```

The container mounts the host dataset at `/app/dataset/data_64Hz` and runs:

```text
python -m src.main
```

## Common Config Knobs

Edit [`src/config.py`](src/config.py) to change:

- dataset path
- split ratios
- window length and stride
- label purity and valid-label thresholds
- ambiguous-window handling
- model type and CNN/LSTM dimensions
- dropout
- batch size
- learning rate
- weight decay
- weighted sampling
- loss selection
- focal-loss gamma and alpha
- early stopping patience

## Troubleshooting

### Dataset directory not found

Set `DREAMT_DATASET_DIR` or place the CSV files under `dataset/data_64Hz`.

### Zero retained windows

Common causes:

- `window_length` is too large for the recordings
- `min_valid_fraction` is too strict
- `label_purity_threshold` is too strict
- `drop_ambiguous_windows` removes too many windows
- timestamp gaps are causing windows to be discarded

### A supervised class is missing from the TRAIN split

The pipeline raises an error if any of `W/N1/N2/N3/R` has zero TRAIN windows because class weights and sampling would be ill-defined.

### Validation macro F1 is still poor

Check:

- `outputs/class_weights.json`
- `outputs/dataset_summary.json`
- `outputs/validation/epochs/`
- `outputs/test/test_per_class_metrics.csv`

The first thing to inspect is usually whether predictions are collapsing into only one or two easier classes.
