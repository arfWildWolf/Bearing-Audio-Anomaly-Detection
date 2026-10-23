# Local bearing anomaly detection

This project trains a compact deep neural network locally to detect bearing
anomalies from log-mel audio spectrogram statistics. It uses normal training
recordings to normalize audio features and a labeled calibration subset to train
the classifier. The remaining labeled test clips are held out for evaluation.

## Setup

From this folder, create and activate a virtual environment, then install the
dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

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

The default run uses normal clips from `bearing/train`. It splits labeled files
in `bearing/test` into a calibration subset and a held-out evaluation subset.
Some calibration clips train the classifier; a separate calibration portion
sets the decision threshold by F1. The evaluation labels are never used for
model fitting or threshold selection. The reported F1 is measured only on the
held-out evaluation subset. This is a supervised detector because anomaly
examples from the calibration split are used during training. Set `--seed` to
reproduce a run.

The trained model and evaluation artifacts are saved under `outputs/`:

- `bearing_classifier.pt` — model weights, normalization values, and threshold
- `metrics.json` — held-out precision, recall, F1, and confusion matrix
- `scores.csv` — per-file anomaly scores, predictions, and split assignments

Use `--data-dir`, `--output-dir`, `--epochs`, or `--evaluation-fraction` to
override defaults. An F1 score of 0.8 or higher is a goal, not a guarantee;
performance depends on the recordings and the held-out split.

## Score new recordings

```powershell
python bearing_anomaly.py predict --checkpoint outputs/bearing_classifier.pt path\to\recording.wav
```

Predictions are printed as JSON lines. The command runs on CPU by default and
uses CUDA when available. When a filename contains `_source_` or `_target_`,
the corresponding calibrated threshold is used; otherwise the overall
threshold is used.
