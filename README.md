# Sleep Stage Classification with DREAMT E4 Signals

This project is a PyTorch sleep stage classifier for the DREAMT 64 Hz aligned wearable dataset. It keeps the existing `python -m src.main` entry point, participant-level split integrity, train-only normalization, diagnostics, metrics, and checkpointing, while using a context-assisted sequence formulation instead of any row-wise classifier behavior.

Each supervised example is one contiguous multichannel time series with shape:

```text
[batch, channels, time]
```

and each example produces one sleep-stage label for the center target segment.

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

If `DREAMT_DATASET_DIR` is not set, the project resolves the dataset directory in this order:

```text
1. dataset/data_64Hz
2. D:\Dreamt\data_64Hz
```

Override either fallback with:

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

## Sequence Windowing and Context

Window generation is configured in `src/config.py`.

Important sequence settings:

- `TARGET_WINDOW_LENGTH`
- `STEP`
- `USE_CONTEXT_WINDOWS`
- `LEFT_CONTEXT`
- `RIGHT_CONTEXT`
- `LABEL_PURITY_THRESHOLD`

Each retained sample is a contiguous input sequence made from:

```text
left context + target segment + right context
```

Default context-assisted setup:

- `TARGET_WINDOW_LENGTH = 256`
- `LEFT_CONTEXT = 256`
- `RIGHT_CONTEXT = 256`
- total model input length = `768`
- the label still comes from the center target segment only

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

Model selection is controlled through `MODEL_NAME` in `src/config.py`.

Supported values:

- `cnn_baseline`
- `cnn_bilstm`
- `cnn_bilstm_target_pool`

The recommended default is `cnn_bilstm_target_pool`.

`cnn_baseline` uses residual 1D convolution blocks and target-aware pooling over the encoded temporal sequence.

`cnn_bilstm_target_pool` uses:

- a Conv1d feature extractor for local temporal structure
- a BiLSTM over the reduced temporal sequence
- explicit target-region pooling over BiLSTM outputs
- optional pooled context summaries outside the target region
- a classifier head for `W`, `N1`, `N2`, `N3`, and `R`

`cnn_bilstm` is kept as a backward-compatible alias of the same CNN+BiLSTM implementation.

Target-aware modeling is explicit:

- target start and end indices are tracked through the pipeline
- target bounds are downsampled through the CNN encoder
- the model appends a target-indicator channel and a normalized relative-position channel to the raw inputs
- pooled summaries can be taken from the target region and, optionally, from the surrounding context region
- context still never changes the assigned target label

The pooling module uses mean, max, and learned attention statistics. When context pooling is enabled, the classifier sees target features, context features, and their difference, which makes the center segment easier to distinguish from surrounding transition context.

## Imbalance Handling and Ablations

Training now exposes loss choice and sampling choice independently so ablations are easy to run from `src/config.py`.

Important knobs:

- `LOSS_NAME`
- `USE_WEIGHTED_SAMPLER`
- `CLASS_WEIGHTING`
- `SAMPLER_WEIGHTING`
- `CLASS_BALANCE_BETA`
- `LABEL_SMOOTHING`
- `focal_gamma`
- `focal_reduction`
- `focal_alpha`

Supported losses:

- `cross_entropy`
- `weighted_cross_entropy`
- `focal_loss`
- `weighted_focal_loss`

Supported class-weighting modes:

- `none`
- `inverse_frequency`
- `sqrt_inverse_frequency`
- `effective_number`

Typical ablations are now straightforward:

- weighted CE only:
  - `LOSS_NAME = "weighted_cross_entropy"`
  - `USE_WEIGHTED_SAMPLER = False`
- focal only:
  - `LOSS_NAME = "focal_loss"`
  - `USE_WEIGHTED_SAMPLER = False`
- weighted sampler only:
  - `LOSS_NAME = "cross_entropy"`
  - `USE_WEIGHTED_SAMPLER = True`
- focal + sampler:
  - `LOSS_NAME = "focal_loss"` or `weighted_focal_loss`
  - `USE_WEIGHTED_SAMPLER = True`
- weighted CE + no sampler:
  - `LOSS_NAME = "weighted_cross_entropy"`
  - `USE_WEIGHTED_SAMPLER = False`

The current defaults are intentionally more conservative than stacking multiple aggressive imbalance tricks:

- `MODEL_NAME = "cnn_bilstm_target_pool"`
- `LOSS_NAME = "weighted_cross_entropy"`
- `USE_WEIGHTED_SAMPLER = False`
- `CLASS_WEIGHTING = "sqrt_inverse_frequency"`
- `SAMPLER_WEIGHTING = "inverse_frequency"`
- `CLASS_BALANCE_BETA = 0.9999`
- `LABEL_SMOOTHING = 0.0`

Class-count and weighting diagnostics are computed from TRAIN target windows only and saved to:

```text
outputs/class_weights.json
```

## Regularization

The training loop keeps practical regularization controls:

- dropout
- weight decay
- label smoothing
- gradient clipping
- early stopping on validation macro F1

Label smoothing is applied to cross-entropy losses and can be disabled easily by setting:

```text
LABEL_SMOOTHING = 0.0
```

For focal losses, label smoothing is not applied.

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
- optional CUDA AMP / mixed precision for both training and evaluation
- optional `ReduceLROnPlateau` scheduling on validation macro F1
- separate training and evaluation batch sizes
- configurable validation artifact frequency
- early stopping on validation macro F1
- best checkpoint selection by validation macro F1
- final test evaluation using the best checkpoint

Runtime diagnostics now log:

- current learning rate
- epoch duration
- training throughput in examples/second
- CUDA peak memory when running on GPU
- batch size, eval batch size, worker count, pin-memory, AMP, and TF32 settings

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
- target distribution per class
- prediction distribution per class
- prediction-minus-target class deltas
- confusion matrices as JSON/CSV/PNG

Per-class CSV outputs now include both:

- true support
- predicted count
- prediction-minus-target count and fraction

This makes it easier to detect distorted prediction frequencies such as minority overprediction or major-class underprediction.

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
- model type, CNN/LSTM dimensions, and target/context fusion behavior
- target indicator and relative-position input channels
- dropout
- batch size and evaluation batch size
- learning rate
- weight decay
- weighted sampling and sampler weighting mode
- loss selection
- class-weighting mode and `effective_number` weighting
- class-balance beta
- focal gamma, alpha, and reduction
- AMP dtype
- deterministic vs benchmarked CUDA behavior
- TF32 usage
- validation artifact frequency
- early stopping patience

## Troubleshooting

### Zero retained windows

Common causes:

- `TARGET_WINDOW_LENGTH` is too large for the recordings
- `LEFT_CONTEXT` and `RIGHT_CONTEXT` make the full input span too large
- `LABEL_PURITY_THRESHOLD` is too strict
- `drop_ambiguous_windows` removes too many target windows
- timestamp gaps are causing windows to be discarded

### Predictions are too biased toward minority classes

Try a less aggressive combination:

- disable `USE_WEIGHTED_SAMPLER`
- use `CLASS_WEIGHTING = "sqrt_inverse_frequency"` instead of full inverse weighting
- compare `CLASS_WEIGHTING = "effective_number"` against the frequency-based modes
- keep `SAMPLER_WEIGHTING` and `CLASS_WEIGHTING` as separate ablations rather than assuming they should match
- compare `weighted_cross_entropy` against `focal_loss`
- check `prediction_minus_target_count` fields in the saved metrics artifacts

### A supervised class is missing from the TRAIN split

The pipeline raises an error if any of `W/N1/N2/N3/R` has zero TRAIN target windows because class weights and sampling would be ill-defined.

### CUDA and throughput checks

When CUDA is active, `outputs/train.log` will show `Using device: cuda` plus the runtime settings for AMP, TF32, batch size, and pin-memory.

Helpful checks:

- run `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"`
- run `nvidia-smi` during training to confirm memory allocation and utilization

### DataLoader workers on Windows

The default `num_workers=0` is conservative on purpose. This project keeps large participant arrays in memory, and higher worker counts on Windows can duplicate that memory because worker processes are spawned instead of forked.

If GPU utilization is low and system RAM is comfortable, you can still try:

- increasing `EVAL_BATCH_SIZE`
- enabling AMP if it was disabled
- raising `num_workers` carefully and monitoring RAM usage
