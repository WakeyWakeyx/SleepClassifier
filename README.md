# Sleep Stage Classification with DREAMT E4 Signals

This project is a PyTorch sleep stage classifier for the DREAMT 64 Hz aligned wearable dataset. It keeps the existing `python -m src.main` entry point, participant-level split integrity, train-only normalization, diagnostics, metrics, and checkpointing, while making the supervised pipeline more explicitly sequence-aware.

The model is not treated as a row-wise classifier. Each supervised example is one contiguous multichannel time series with shape:

```text
[batch, channels, time]
```

and each example produces one sleep-stage label for the target segment.

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

## Sequence Windowing and Target Labels

Window generation is configured in `src/config.py`.

Important sequence settings:

- `TARGET_WINDOW_LENGTH`
- `STEP`
- `USE_CONTEXT_WINDOWS`
- `LEFT_CONTEXT`
- `RIGHT_CONTEXT`
- `LABEL_PURITY_THRESHOLD`

The underlying dataclass fields are:

- `config.data.target_window_length`
- `config.data.step`
- `config.data.use_context_windows`
- `config.data.left_context`
- `config.data.right_context`
- `config.data.label_purity_threshold`

Each retained sample is a contiguous time sequence made from:

```text
left context + target segment + right context
```

Examples:

- standard sequence classification:
  - `USE_CONTEXT_WINDOWS = False`
  - model input = target segment only
  - label = target segment only
- context-assisted sequence classification:
  - `TARGET_WINDOW_LENGTH = 256`
  - `LEFT_CONTEXT = 256`
  - `RIGHT_CONTEXT = 256`
  - total model input length = `768`
  - label still comes from the center target segment only

Important label behavior:

- `P` and `Missing` are excluded from supervised targets
- majority vote is computed on the target segment only
- purity checks are computed on the target segment only
- ambiguous target windows can still be dropped
- surrounding context never changes the assigned target label
- windows spanning large timestamp discontinuities are dropped

To avoid partial-context edge cases, context-assisted samples are only generated when the full input span fits within the participant file boundaries.

Window diagnostics are saved after generation for train, validation, and test, including:

- per-class retained target-window counts
- raw candidate windows per file
- kept windows per file
- discarded windows per file
- discard reasons such as temporal gaps, excluded-label contamination, and insufficient purity
- sequence-definition metadata such as target length, left context, right context, and total input length

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
- class counts and class weights: TRAIN target windows only
- weighted sampling: TRAIN target windows only

## Model Options

Model selection is controlled through `MODEL_NAME` in `src/config.py` and exposed as `config.model.model_name`.

Supported values:

- `cnn_baseline`
- `cnn_bilstm`

`cnn_baseline` uses residual 1D convolution blocks to extract temporal features, then pools only over the target region representation.

`cnn_bilstm` uses the same temporal CNN stem, converts features to `[batch, seq_len, feature_dim]`, runs a bidirectional LSTM over time, and then pools only over the target-region outputs.

This is the key target-region behavior:

- context can help the CNN or BiLSTM build better temporal features
- the final classifier does not blindly pool over the whole input
- the final pooled representation is computed from the target region only

## Imbalance Handling and Loss Options

Training supports configurable class-imbalance handling from `src/config.py`:

- `USE_WEIGHTED_SAMPLER`
- `LOSS_NAME`
- `LABEL_SMOOTHING`
- `CLASS_WEIGHTING`
- `focal_gamma`
- `focal_use_class_weights`
- `focal_alpha`

The dataclass fields are:

- `config.training.use_weighted_sampler`
- `config.training.loss_name`
- `config.training.label_smoothing`
- `config.training.class_weighting_mode`

Supported losses:

- `cross_entropy`
- `weighted_cross_entropy`
- `focal_loss`

Supported class-weighting modes:

- `inverse_frequency`
- `none`

Class weights are computed from TRAIN target windows only and saved to:

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

If context windows are enabled, training logs include:

- target length
- left context
- right context
- total input length

## Evaluation Diagnostics

Validation and test evaluation save rich diagnostics under `outputs/`.

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

These artifacts are intended to make class-collapse and temporal-context issues easier to inspect, especially for `N3` and `R`.

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

Edit `src/config.py` to change:

- dataset path
- split ratios
- target window length and stride
- context window lengths
- label purity and valid-label thresholds
- ambiguous-window handling
- model type and CNN/LSTM dimensions
- dropout
- batch size
- learning rate
- weight decay
- weighted sampling
- loss selection
- class-weighting mode
- focal-loss gamma and alpha
- early stopping patience

## Troubleshooting

### Dataset directory not found

Set `DREAMT_DATASET_DIR` or place the CSV files under `dataset/data_64Hz`.

### Zero retained windows

Common causes:

- `TARGET_WINDOW_LENGTH` is too large for the recordings
- `LEFT_CONTEXT` and `RIGHT_CONTEXT` make the full input span too large
- `LABEL_PURITY_THRESHOLD` is too strict
- `drop_ambiguous_windows` removes too many target windows
- timestamp gaps are causing windows to be discarded

### A supervised class is missing from the TRAIN split

The pipeline raises an error if any of `W/N1/N2/N3/R` has zero TRAIN target windows because class weights and sampling would be ill-defined.

### Validation macro F1 is still poor

Check:

- `outputs/class_weights.json`
- `outputs/dataset_summary.json`
- `outputs/validation/epochs/`
- `outputs/test/test_per_class_metrics.csv`

The first thing to inspect is usually whether predictions are collapsing into only one or two easier classes, or whether the target window is still too short to capture enough temporal context for `N3` and `R`.
