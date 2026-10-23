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

The VS Code task **Train Bearing Anomaly Detector** explicitly requests CUDA.
The CLI defaults to `--device auto`, which uses CUDA when available; use
`--device cuda` to require a GPU (it errors rather than silently falling back),
or `--device cpu` to force CPU execution.

The default run uses normal clips from `bearing/train`. It splits labeled files
in `bearing/test` into a calibration subset and a held-out evaluation subset.
Some calibration clips train the classifier; a separate calibration portion
sets the threshold to maximize recall while maintaining precision of at least
0.70. Override the precision target with `--minimum-precision`. Threshold
selection uses calibration labels only; the evaluation labels are never used
for model fitting or threshold selection. The held-out report includes
precision and recall, which may differ from the calibration target due to
sampling variation. This is a supervised detector because anomaly examples
from the calibration split are used during training. Set `--seed` to reproduce
a run.

The trained model and evaluation artifacts are saved under `outputs/`:

- `bearing_classifier.pt` — model weights, normalization values, and threshold
- `metrics.json` — held-out precision, recall, F1, and confusion matrix
- `scores.csv` — per-file anomaly scores, predictions, and split assignments

Use `--data-dir`, `--output-dir`, `--epochs`, `--evaluation-fraction`, or
`--minimum-precision` to override defaults. Performance depends on the
recordings and held-out split; a calibration precision constraint cannot
guarantee the same precision on future data.

## Score new recordings

```powershell
python bearing_anomaly.py predict --device cuda --checkpoint outputs/bearing_classifier.pt path\to\recording.wav
```

Predictions are printed as JSON lines. `--device auto` runs on CUDA when
available and otherwise falls back to CPU. With `--device cuda`, it requires
CUDA and reports an error rather than falling back. When a filename contains
`_source_` or `_target_`, the corresponding calibrated threshold is used;
otherwise the overall threshold is used.
