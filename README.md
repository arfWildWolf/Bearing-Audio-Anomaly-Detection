# Local bearing anomaly detection

This project trains a compact deep neural network locally to detect bearing
anomalies from log-mel audio spectrogram statistics. It uses normal training
recordings as labeled normal examples and uses a labeled calibration subset to
train the classifier. A separate labeled subset calibrates its threshold, and
the remaining test clips are held out for evaluation.

## Setup

From this folder, create and activate a virtual environment, then install the
dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0+cu128"
```

The second command installs the CUDA 12.8 PyTorch build for NVIDIA GPUs
(including the RTX 50 series). Ensure a recent NVIDIA driver is installed.
To verify it, run `python gpu_check.py`; this project has been tested with
PyTorch 2.11.0+cu128 on an RTX 5060 Laptop GPU.

The default data layout is:

```text
bearing/
  train/   # normal WAV files
  test/    # WAV names include _normal_ or _anomaly_
```

## Train and evaluate

```powershell
python bearing_anomaly.py train
```

To compare training with and without the extra normal training recordings, as
well as the larger classifier, under the same 5% calibration false-positive
rate cap, use:

```powershell
python bearing_anomaly.py compare --device cuda
```

This compares three candidates on stratified folds drawn only from the
calibration partition and keeps the final evaluation partition untouched. It
selects between the small classifier trained only on labeled calibration clips
and the same classifier additionally trained with the normal recordings,
preferring the highest-recall strategy whose pooled validation false-positive
rate is within the configured cap. The larger model is selected only if it
beats the better small-model strategy on recall in at least 80% of folds,
without increasing false positives, and stays within the same pooled
false-positive-rate cap. Only then is a 30%-width structured-pruned model
evaluated; pruning is selected only if its recall gain over that small-model
baseline survives at no greater false-positive count and within the cap. The
selected model is then trained and evaluated on the reserved holdout.

The VS Code task **Train Bearing Anomaly Detector** explicitly requests CUDA.
The CLI defaults to `--device auto`, which uses CUDA when available; use
`--device cuda` to require a GPU (it errors rather than silently falling back),
or `--device cpu` to force CPU execution.

The default run uses normal clips from `bearing/train`. It splits labeled files
in `bearing/test` into a calibration subset and a held-out evaluation subset.
Some calibration clips train the classifier; a separate calibration portion
sets thresholds to maximize recall while limiting the calibration false-positive
rate to 5% (`--maximum-false-positive-rate`), with separate thresholds for the
`source` and `target` domains when identified in the filename. Set the cap to
`0` to allow no false positives on each calibration domain. Threshold
selection uses calibration labels only; held-out evaluation
labels do not affect the model or threshold. The report includes held-out
false-positive count and rate, which can differ from calibration because of
sampling variation. This is a supervised detector because anomaly examples
from the calibration split are used during training. Set `--seed` to reproduce
a run.

The trained model and evaluation artifacts are saved under `outputs/`:

- `bearing_classifier.pt` — model weights, normalization values, and threshold
- `metrics.json` — held-out precision, recall, F1, and confusion matrix
- `scores.csv` — per-file anomaly scores, predictions, and split assignments
- `model_comparison.json` — cross-validation results for each model candidate
- `cross_validation_false_positives.csv` — normal validation clips flagged by
  each candidate, with their scores and fold thresholds
- `false_positive_review.csv` — held-out and threshold-validation normal clips
  sorted by distance above/below the decision threshold

Use `--data-dir`, `--output-dir`, `--epochs`, `--evaluation-fraction`, or
`--maximum-false-positive-rate` to override defaults. A strict false-positive
cap may reduce recall, and the calibration cap cannot guarantee the same
false-positive rate on future recordings.

## Score new recordings

```powershell
python bearing_anomaly.py predict --device cuda --checkpoint outputs/bearing_classifier.pt path\to\recording.wav
```

Predictions are printed as JSON lines. `--device auto` runs on CUDA when
available and otherwise falls back to CPU. With `--device cuda`, it requires
CUDA and reports an error rather than falling back. The detector applies the
same false-positive-conscious threshold to each recording.
