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

To compare classifier and score-normalization strategies using standardized
partial ROC AUC in the low-false-positive region, use:

```powershell
python bearing_anomaly.py compare --device cuda
```

The comparison uses stratified folds drawn only from the calibration partition;
the final evaluation partition remains untouched. Candidates include the small
MLP with and without the extra normal training clips, the larger MLP, and an
RBF support-vector machine. Raw and k-nearest-normal-distance-adjusted scores
are both tested, then converted to empirical ranks against normal clips in each
fold's threshold-calibration subset. The comparison also tests a small-model
candidate that concatenates the log-mel summary with mean and standard
deviation embeddings from the locally available convolutional autoencoder
before distance scoring. Its two feature blocks are standardized
and balanced so the larger log-mel block does not overwhelm the latent block.
Separate normal-score references are maintained for source and target domains
when available, with the pooled reference as a fallback. The MLP and distance
adjustment run on the selected device; SVM fitting runs locally on CPU.

This local ensemble uses the saved autoencoder checkpoint at
`outputs/bearing_autoencoder.pt`; set `--autoencoder-checkpoint` to override
its path. It is an in-domain autoencoder representation, not a BEATs or EAT
foundation-model embedding. Those pretrained models are not included in the
workspace.

For paired near/far stereo recordings, train the dual-channel candidate with:

```powershell
python bearing_anomaly.py train --architecture small_dual_channel_normal_rank --data-dir path\to\stereo-bearing
```

The candidate requires exactly two channels in every WAV, with channel 1
(`ch01`) near-field and channel 2 (`ch02`) far-field. It builds separate
640-dimensional log-mel summaries and trains the MLP on
`[near, near - far]` (1280 inputs), then applies the same empirical normal-rank
calibration and thresholding as the mono model. Prediction with its checkpoint
also requires two-channel WAVs in the same channel order. Mono data cannot
measure this candidate; the existing mono baseline remains unchanged.

The comparison reports standardized partial ROC AUC with maximum FPR 0.1, and
selects the best strategy by mean fold harmonic mean (F1). Larger models must
improve F1 in at least 80% of folds and improve mean F1 to qualify; a
structured-pruned candidate is tested only if the larger MLP qualifies, and is
retained only if its improvement survives. pAUC remains a reported secondary
ranking metric. The winning candidate is retrained and evaluated on the
reserved holdout. Operating thresholds are independently calibrated per
domain to maximize recall while keeping the empirical calibration-normal FPR
strictly below `--maximum-false-positive-rate` (default 0.1). This empirical
calibration constraint is not a guarantee for future data. Partial AUC measures
score ranking over the low-FPR region; it is not optimized by a single
operating threshold.

The VS Code task **Train Bearing Anomaly Detector** explicitly requests CUDA.
The CLI defaults to `--device auto`, which uses CUDA when available; use
`--device cuda` to require a GPU (it errors rather than silently falling back),
or `--device cpu` to force CPU execution.

The default run uses normal clips from `bearing/train`. It splits labeled files
in `bearing/test` into a calibration subset and a held-out evaluation subset.
Some calibration clips train the classifier; a separate calibration portion
provides the normal-score reference and operating threshold. The default
`small_normal_rank` architecture maps scores to empirical normal-validation
ranks before thresholding. Domain-specific references and thresholds are used
when calibration normals from that domain are present; otherwise the pooled
normal reference and threshold are used. Threshold selection uses calibration
labels only; held-out evaluation labels do not affect the model, normalization,
or threshold. Set `--maximum-false-positive-rate` to change the strict
calibration FPR cap.

When `compare` selects the autoencoder feature-stack candidate, the saved
checkpoint contains its encoder and normalization values so prediction applies
the same feature transform locally.

The report includes thresholded metrics (accuracy, balanced accuracy,
precision, recall/sensitivity, specificity, negative predictive value,
false-positive and false-negative rates, F1, F2, the precision-recall harmonic
mean, Matthews correlation coefficient, and confusion-matrix counts) as well
as ROC AUC, standardized partial ROC AUC at FPR <= 0.1, and average precision
(PR AUC). The harmonic mean of precision and recall is mathematically F1, so
those two values should agree apart from rounding. AUC metrics evaluate score
ranking and do not depend on the selected decision threshold. Metrics are
reported for both threshold-validation and held-out evaluation sets and may
differ because of sampling variation. The calibration FPR cap is empirical,
not a guarantee for future recordings. This is a supervised detector because
anomaly examples from the calibration split are used during training. Set
`--seed` to reproduce a run.

The trained model and evaluation artifacts are saved under `outputs/`:

- `bearing_classifier.pt` — model weights, normalization values, normal-rank
  references, and calibrated thresholds
- `autoencoder_feature_stack_candidate/` — separately trained ensemble
  checkpoint and held-out metrics for the raw-plus-autoencoder KNN experiment
- `metrics.json` — thresholded metrics and score-based ROC/pAUC/PR AUC for
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

Predictions are printed as JSON lines; rank-normalized models emit an
`anomaly_score` with `score_kind: empirical normal-validation rank` rather than
a calibrated probability. `--device auto` runs on CUDA when
available and otherwise falls back to CPU. With `--device cuda`, it requires
CUDA and reports an error rather than falling back. The detector applies the
threshold calibrated under the configured maximum-FPR constraint.
