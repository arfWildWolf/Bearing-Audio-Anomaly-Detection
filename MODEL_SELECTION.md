# Model selection rationale

This note records the selection criteria, evidence, and trade-offs behind the
current bearing anomaly detector. It summarizes the decision from experiment
artifacts; it is not a transcript of private chain-of-thought.

## Current selection

The current selected checkpoint is **`small_normal_rank`**, a small supervised
MLP with empirical normal-score rank calibration. It was selected under the
project's cross-validation rules, not because it had the highest raw mean F1
among every model tested.

The selection process compares models only inside a calibration partition.
It reserves a separate labeled holdout for final evaluation. The current run
uses five stratified folds, and selects among eligible candidates by mean
cross-validated F1. Ties are broken by fewer false positives.

## Why this evaluation design

- **Use the labeled anomaly examples.** The initial unsupervised autoencoder
  did not meet the target reliably, so the workflow moved to a supervised
  classifier trained with labeled calibration anomalies and normal training
  recordings.
- **Keep the final holdout out of model selection.** The labeled test set is
  split into calibration data and a final evaluation subset. Candidate
  training, rank calibration, and threshold selection use calibration data;
  held-out labels are used only for the final report.
- **Treat false positives as a constraint.** The desired operating threshold
  maximizes recall while keeping the observed calibration-normal false
  positive rate strictly below the configured cap (0.1 by default). This is
  an empirical calibration constraint, not a guarantee about future clips.
- **Normalize scores against normal clips.** Raw score scales can differ
  across domains. The pipeline maps scores to empirical ranks against
  calibration normals, keeps source/target normal references separately
  where available, and uses the pooled reference as a fallback.
- **Report ranking separately from thresholded decisions.** ROC AUC,
  standardized pAUC at FPR <= 0.1, and average precision measure score ranking.
  F1, precision, recall, and specificity measure decisions at the calibrated
  threshold. Improving one does not ensure improvement in the other.

## Candidate comparison

The comparison evaluated five-fold cross-validation in the calibration
partition. Each fold calibrated its threshold independently. The following
values come from `outputs/model_comparison.json`; “fold FP” is the sum of
false positives across fold evaluations.

| Candidate | Mean fold F1 | Pooled F1 | Mean pAUC@0.1 | Fold FP |
|---|---:|---:|---:|---:|
| Small MLP, calibration examples only | 0.386 | 0.404 | 0.553 | 9 |
| Small MLP, calibration only + kNN score adjustment | 0.428 | 0.440 | 0.561 | 8 |
| Small MLP + normal training examples | **0.493** | 0.531 | 0.698 | 2 |
| Small MLP + kNN score adjustment | 0.471 | 0.505 | 0.698 | 1 |
| Small MLP + kNN and autoencoder features | 0.470 | 0.505 | **0.720** | 1 |
| Large MLP | 0.498 | 0.529 | 0.655 | 5 |
| Large MLP + kNN score adjustment | 0.513 | 0.544 | 0.658 | 5 |
| RBF SVM | 0.607 | 0.618 | 0.649 | 6 |
| RBF SVM + kNN score adjustment | 0.619 | **0.631** | 0.637 | 6 |

### Why the small MLP was the eligible baseline

Among the small-model training strategies, adding the 1,000 normal training
clips gave the highest mean fold F1 (0.493), with only two fold-evaluation
false positives. The kNN and autoencoder variants reduced the summed false
positives to one and slightly improved pAUC for the autoencoder variant, but
both had lower mean F1. Since F1 was the primary selection metric, those were
not selected.

The calibration-only candidates also had lower mean F1. This supports using
the additional normal training clips as negative classifier examples rather
than training only from labeled calibration examples.

### Why the large MLP and SVM did not replace it

A larger architecture or a higher average score was not enough by itself.
The larger MLP and SVM candidates had to beat the small-model baseline in at
least 80% of folds and improve mean fold F1. With five folds, that requires
four wins.

- The large MLP won three folds, below the four-fold requirement.
- The RBF SVM also won three folds. Its mean F1 was higher, but its results
  were not consistent enough to qualify under the pre-set rule.
- The kNN-adjusted large MLP and kNN-adjusted SVM each won three folds and
  also failed the consistency requirement.
- Because no large MLP candidate qualified, structured pruning was skipped.
  Pruning is only tested after consistent evidence that the larger model
  improves; it is retained only if the improvement survives pruning.

This rule favors repeatable improvements across validation splits over a
single high mean that may be driven by a subset of folds. It is a deliberate
stability criterion, not proof that the small MLP has the best possible
performance.

## Why `normal_rank` is part of the selected architecture

The rank stage converts each model's score into its relative position among
normal calibration clips. Domain-specific normal references and thresholds
are used when enough calibration data exists, with pooled calibration as a
fallback. This makes the score scale less dependent on raw output magnitude
and supports threshold calibration under the low-FPR constraint.

The ranking transform does not guarantee domain-shift immunity. The observed
FPR constraint is estimated from a finite calibration sample and may not
generalize to new machines or recording conditions.

## Current held-out result

The selected `small_normal_rank` checkpoint's recorded held-out evaluation in
`outputs/metrics.json` contains 60 clips: 30 normal and 30 anomalous.

| Metric | Held-out result |
|---|---:|
| F1 / precision-recall harmonic mean | 0.400 |
| Precision | 0.800 |
| Recall | 0.267 |
| Specificity | 0.933 |
| Accuracy | 0.600 |
| ROC AUC | 0.781 |
| Standardized pAUC, FPR <= 0.1 | 0.605 |
| Average precision | 0.751 |
| False positives | 2 |
| False negatives | 22 |

The corresponding confusion matrix is 28 true negatives, 2 false positives,
22 false negatives, and 8 true positives. The held-out FPR is 2/30 (0.067).
Threshold-validation F1 is higher (0.732), but it is not the final performance
estimate; use the held-out results when describing generalization.

The operating threshold is calibrated on threshold-validation clips and is
not selected by maximizing held-out F1. This separation avoids tuning the
reported final score to the evaluation labels.

## Other feature experiments and limitations

- **Autoencoder feature stack:** the locally available convolutional
  autoencoder was tested as an additional kNN feature block. In its separate
  held-out report, F1 remained 0.400, while ROC AUC (0.769) and pAUC@0.1
  (0.579) were lower than the current selected checkpoint. The CV autoencoder
  candidate had the highest mean pAUC among the small variants, but did not
  win on the primary F1 criterion.
- **BEATs/EAT embeddings:** the workspace did not include these models or
  weights, so they were not benchmarked.
- **Dual-channel near/far features:** the `[near, near - far]` candidate was
  implemented and smoke-tested with synthetic stereo audio. The bundled
  recordings are mono, so no real-data training or comparison metrics exist
  for this candidate. Synthetic smoke-test metrics are not evidence of
  bearing-dataset performance.

## Reproducibility and caveats

Use the current `outputs/metrics.json` and `outputs/model_comparison.json` as
the evidence for this selection. Earlier experiments used different
threshold objectives, folds, or random seeds; their scores should not be
compared as if they were one controlled run. The current run records seed 42
and CUDA training. Small sample counts, particularly in the threshold
calibration subsets, make both F1 and FPR estimates uncertain.

To reproduce the comparison, run:

```powershell
python bearing_anomaly.py compare --device cuda
```

The result should be treated as the best candidate under the recorded
comparison protocol, not as a claim that the model meets the original F1
target or is universally optimal.
