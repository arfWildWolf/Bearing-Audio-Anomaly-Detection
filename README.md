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
well as the larger classifier, by cross-validated F1, use:

```powershell
python bearing_anomaly.py compare --device cuda
```

This compares four candidates on stratified folds drawn only from the
calibration partition and keeps the final evaluation partition untouched.
The candidates are the small MLP with and without the extra normal recordings,
the larger MLP, and an RBF support-vector machine. MLP training and scoring
use CUDA when requested; the scikit-learn SVM fit runs locally on CPU and its
scoring is performed on the selected device. Thresholds are selected to
maximize F1 on a separate calibration subset, with
separate thresholds for `source` and `target` when both classes are represented
in that domain. The small-model strategy with the highest pooled cross-validated
F1 is selected as the baseline (breaking ties toward fewer false positives).
The larger MLP and SVM are selected only if each improves F1 over the baseline
in at least 80% of folds and improves pooled F1. Only when the larger MLP meets
that rule is a 30%-width structured-pruned model evaluated; pruning is selected
only if its F1 improvement over the small baseline also survives in at least
80% of folds and improves pooled F1. The best qualifying candidate is trained
and evaluated on the reserved holdout. No false-positive-rate cap is applied
in this F1-optimization mode; precision, recall, and false positives remain
reported for transparency. The SVM's sigmoid-transformed decision score is
monotonic but is not a calibrated probability.

The VS Code task **Train Bearing Anomaly Detector** explicitly requests CUDA.
The CLI defaults to `--device auto`, which uses CUDA when available; use
`--device cuda` to require a GPU (it errors rather than silently falling back),
or `--device cpu` to force CPU execution.

The default run uses normal clips from `bearing/train`. It splits labeled files
in `bearing/test` into a calibration subset and a held-out evaluation subset.
Some calibration clips train the classifier; a separate calibration portion
sets thresholds to maximize F1, with separate thresholds for the `source` and
`target` domains when identified in the filename. Threshold selection uses
calibration labels only; held-out evaluation labels do not affect the model or
threshold. The report includes thresholded metrics (accuracy, balanced accuracy,
precision, recall/sensitivity, specificity, negative predictive value,
false-positive and false-negative rates, F1, the precision-recall harmonic
mean, Matthews correlation coefficient, and confusion-matrix counts) as well
as score-based ROC AUC and average precision (PR AUC). The harmonic mean of
precision and recall is mathematically F1, so those two values should agree
apart from rounding. AUC metrics evaluate score ranking and do not depend on
the selected decision threshold. Metrics are reported for both threshold
validation and held-out evaluation, and may differ because of sampling
variation. F1 optimization does not guarantee a low false-positive rate. This
is a supervised detector because anomaly examples from the calibration split
are used during training. Set `--seed` to reproduce a run.

The trained model and evaluation artifacts are saved under `outputs/`:

- `bearing_classifier.pt` — model weights, normalization values, and threshold
- `metrics.json` — thresholded metrics and score-based ROC/PR AUC for
  threshold-validation and held-out evaluation sets
- `scores.csv` — per-file anomaly scores, predictions, and split assignments
- `model_comparison.json` — cross-validation results for each model candidate
- `cross_validation_false_positives.csv` — normal validation clips flagged by
  each candidate, with their scores and fold thresholds
- `false_positive_review.csv` — held-out and threshold-validation normal clips
  sorted by distance above/below the decision threshold

Use `--data-dir`, `--output-dir`, `--epochs`, or `--evaluation-fraction` to
override defaults.

## Score new recordings

```powershell
python bearing_anomaly.py predict --device cuda --checkpoint outputs/bearing_classifier.pt path\to\recording.wav
```

Predictions are printed as JSON lines. `--device auto` runs on CUDA when
available and otherwise falls back to CPU. With `--device cuda`, it requires
CUDA and reports an error rather than falling back. The detector applies the
threshold selected to maximize F1 on the calibration subset.
