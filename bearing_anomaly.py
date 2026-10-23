from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


SAMPLE_RATE = 16_000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
N_FRAMES = 128


def _hz_to_mel(frequency: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + frequency / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank() -> torch.Tensor:
    mel_points = np.linspace(
        _hz_to_mel(np.array([0.0]))[0],
        _hz_to_mel(np.array([SAMPLE_RATE / 2]))[0],
        N_MELS + 2,
    )
    hz_points = _mel_to_hz(mel_points)
    fft_frequencies = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE)
    filters = np.zeros((N_MELS, len(fft_frequencies)), dtype=np.float32)

    for band in range(N_MELS):
        left, center, right = hz_points[band : band + 3]
        filters[band] = np.maximum(
            0.0,
            np.minimum(
                (fft_frequencies - left) / max(center - left, 1e-12),
                (right - fft_frequencies) / max(right - center, 1e-12),
            ),
        )
    return torch.from_numpy(filters)


MEL_FILTERBANK = _mel_filterbank()


def extract_feature(path: Path) -> np.ndarray:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if waveform.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    if sample_rate != SAMPLE_RATE:
        divisor = np.gcd(sample_rate, SAMPLE_RATE)
        waveform = resample_poly(
            waveform, SAMPLE_RATE // divisor, sample_rate // divisor
        ).astype(np.float32)

    audio = torch.from_numpy(waveform.copy())
    spectrum = torch.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=torch.hann_window(N_FFT),
        center=True,
        return_complex=True,
    ).abs()
    mel_power = MEL_FILTERBANK @ spectrum.square()
    log_mel = torch.log(mel_power + 1e-8)
    resized = F.adaptive_avg_pool1d(log_mel.unsqueeze(0), N_FRAMES).squeeze(0)
    return resized.numpy().astype(np.float32)


def load_features(paths: list[Path]) -> np.ndarray:
    if not paths:
        raise ValueError("No WAV files were found.")
    return np.stack([extract_feature(path) for path in paths])


def labels_from_paths(paths: list[Path]) -> np.ndarray:
    labels = []
    for path in paths:
        name = path.name.lower()
        if "_anomaly_" in name:
            labels.append(1)
        elif "_normal_" in name:
            labels.append(0)
        else:
            raise ValueError(
                f"Cannot infer normal/anomaly label from test filename: {path.name}"
            )
    return np.asarray(labels, dtype=np.int64)


def stratification_labels(paths: list[Path], labels: np.ndarray) -> np.ndarray:
    groups = []
    for path, label in zip(paths, labels):
        name = path.name.lower()
        if "_source_" in name:
            domain = "source"
        elif "_target_" in name:
            domain = "target"
        else:
            domain = "all"
        groups.append(f"{domain}_{label}")
    return np.asarray(groups)


def _domain(path: Path) -> str | None:
    name = path.name.lower()
    if "_source_" in name:
        return "source"
    if "_target_" in name:
        return "target"
    return None


class FeatureClassifier(nn.Module):
    def __init__(self, hidden_dims: tuple[int, ...] = (256, 64)) -> None:
        super().__init__()
        self.hidden_dims = hidden_dims
        layers: list[nn.Module] = []
        input_dim = N_MELS * 10
        for index, hidden_dim in enumerate(hidden_dims):
            layers.append(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.3 if index == 0 else 0.2),
                )
            )
            input_dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            values = layer(values)
        return self.classifier(values).squeeze(1)


ARCHITECTURES: dict[str, tuple[int, ...]] = {
    "small": (256, 64),
    "large": (512, 256, 128),
}


def _prune_mlp(model: FeatureClassifier, keep_fraction: float) -> FeatureClassifier:
    if not 0 < keep_fraction < 1:
        raise ValueError("Pruning keep fraction must be between 0 and 1.")
    hidden_dims = tuple(
        max(8, int(round(width * keep_fraction))) for width in model.hidden_dims
    )
    pruned = FeatureClassifier(hidden_dims).to(next(model.parameters()).device)
    previous_indices = torch.arange(
        N_MELS * 10, device=next(model.parameters()).device
    )

    with torch.no_grad():
        for old_layer, new_layer, new_width in zip(
            model.layers, pruned.layers, hidden_dims
        ):
            old_linear, old_norm = old_layer[0], old_layer[1]
            new_linear, new_norm = new_layer[0], new_layer[1]
            importance = old_linear.weight.abs().sum(dim=1)
            selected = importance.topk(new_width, sorted=False).indices
            new_linear.weight.copy_(
                old_linear.weight[selected][:, previous_indices]
            )
            new_linear.bias.copy_(old_linear.bias[selected])
            new_norm.weight.copy_(old_norm.weight[selected])
            new_norm.bias.copy_(old_norm.bias[selected])
            previous_indices = selected
        pruned.classifier.weight.copy_(
            model.classifier.weight[:, previous_indices]
        )
        pruned.classifier.bias.copy_(model.classifier.bias)
    return pruned


def _prepare_model_features(raw: np.ndarray, architecture: str) -> np.ndarray:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown model architecture: {architecture}")
    return _summarize_spectrograms(raw)


def _probabilities(
    model: FeatureClassifier, features: np.ndarray, device: torch.device
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), 64):
            batch = torch.from_numpy(features[start : start + 64]).to(device)
            probabilities.append(
                torch.sigmoid(model(batch)).cpu().numpy()
            )
    return np.concatenate(probabilities)


def _maximum_f1_threshold(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[float, int]:
    candidates = np.concatenate(
        [np.nextafter(np.max(scores), np.inf).reshape(1), np.unique(scores)]
    )
    candidates_by_f1: list[tuple[float, int, float]] = []
    for threshold in candidates:
        predictions = scores >= threshold
        false_positives = int(np.count_nonzero(predictions & (labels == 0)))
        true_positives = int(np.count_nonzero(predictions & (labels == 1)))
        false_negatives = int(np.count_nonzero(~predictions & (labels == 1)))
        denominator = 2 * true_positives + false_positives + false_negatives
        f1 = 2 * true_positives / denominator if denominator else 0.0
        candidates_by_f1.append((f1, false_positives, float(threshold)))
    f1, false_positives, threshold = max(
        candidates_by_f1, key=lambda result: (result[0], -result[1], result[2])
    )
    return threshold, false_positives


def _calibrate_thresholds(
    paths: list[Path],
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, dict[str, float], int]:
    overall_threshold, overall_false_positives = _maximum_f1_threshold(labels, scores)
    thresholds: dict[str, float] = {}
    domain_false_positives = 0
    for domain in ("source", "target"):
        mask = np.array([_domain(path) == domain for path in paths])
        if not mask.any() or not np.any(labels[mask] == 0) or not np.any(labels[mask] == 1):
            continue
        thresholds[domain], false_positives = _maximum_f1_threshold(
            labels[mask], scores[mask]
        )
        domain_false_positives += false_positives
    return (
        overall_threshold,
        thresholds,
        domain_false_positives
        if thresholds
        else overall_false_positives,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def _get_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but PyTorch cannot access a CUDA GPU. "
                "Install a CUDA-enabled PyTorch build and verify the NVIDIA driver."
            )
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize(
    features: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return np.clip(
        (features - mean[None, None, :, None]) / std[None, None, :, None],
        -8.0,
        8.0,
    ).astype(np.float32)


def _summarize_spectrograms(features: np.ndarray) -> np.ndarray:
    spectrograms = features[:, 0]
    delta = np.diff(spectrograms, axis=2)
    chunks = np.array_split(spectrograms, 4, axis=2)
    summaries = [
        spectrograms.mean(axis=2),
        spectrograms.std(axis=2),
        np.quantile(spectrograms, 0.1, axis=2),
        np.quantile(spectrograms, 0.9, axis=2),
        np.abs(delta).mean(axis=2),
        delta.std(axis=2),
        *(chunk.mean(axis=2) for chunk in chunks),
    ]
    return np.concatenate(summaries, axis=1).astype(np.float32)


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negatives, false_positives = matrix[0]
    return {
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "false_positives": int(false_positives),
        "true_positives": int(matrix[1, 1]),
        "false_positive_rate": float(
            false_positives / (true_negatives + false_positives)
            if true_negatives + false_positives
            else 0.0
        ),
        "confusion_matrix_normal_anomaly": matrix.tolist(),
        "samples": int(len(labels)),
        "anomalies": int(labels.sum()),
    }


def _fit_classifier(
    labeled_features: np.ndarray,
    labels: np.ndarray,
    normal_features: np.ndarray,
    architecture: str,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    prune_keep_fraction: float | None = None,
) -> FeatureClassifier:
    _set_seed(seed)
    train_indices, validation_indices = train_test_split(
        np.arange(len(labels)),
        test_size=0.15,
        random_state=seed,
        stratify=labels,
    )
    labeled_train = torch.from_numpy(labeled_features[train_indices])
    labels_train = torch.from_numpy(labels[train_indices].astype(np.float32))
    validation = torch.from_numpy(labeled_features[validation_indices]).to(device)
    validation_labels = torch.from_numpy(
        labels[validation_indices].astype(np.float32)
    ).to(device)
    normal_train = torch.from_numpy(normal_features)

    train_features = torch.cat([normal_train, labeled_train], dim=0)
    train_labels = torch.cat(
        [torch.zeros(len(normal_train), dtype=torch.float32), labels_train], dim=0
    )
    positive_count = int(train_labels.sum().item())
    negative_count = len(train_labels) - positive_count
    sample_weights = torch.where(
        train_labels > 0,
        1.0 / max(positive_count, 1),
        1.0 / max(negative_count, 1),
    )
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_labels),
        replacement=True,
    )
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
    )

    model = FeatureClassifier(ARCHITECTURES[architecture]).to(device)
    loss_function = nn.BCEWithLogitsLoss()

    def fit_stage(stage_model: FeatureClassifier) -> FeatureClassifier:
        optimizer = torch.optim.AdamW(
            stage_model.parameters(), lr=learning_rate, weight_decay=1e-3
        )
        best_loss = float("inf")
        best_state = copy.deepcopy(stage_model.state_dict())
        epochs_without_improvement = 0
        stage_model.train()
        for _ in range(epochs):
            stage_model.train()
            for batch, batch_labels in loader:
                batch = batch.to(device)
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(stage_model(batch), batch_labels)
                loss.backward()
                optimizer.step()

            stage_model.eval()
            with torch.no_grad():
                validation_loss = float(
                    loss_function(stage_model(validation), validation_labels).item()
                )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = copy.deepcopy(stage_model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break
        stage_model.load_state_dict(best_state)
        return stage_model

    model = fit_stage(model)
    if prune_keep_fraction is not None:
        model = _prune_mlp(model, prune_keep_fraction)
        model = fit_stage(model)
    return model


def train(args: argparse.Namespace) -> dict[str, object]:
    _set_seed(args.seed)
    data_dir = args.data_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_paths = sorted((data_dir / "train").glob("*.wav"))
    test_paths = sorted((data_dir / "test").glob("*.wav"))
    if not train_paths or not test_paths:
        raise FileNotFoundError(
            f"Expected WAV files in {data_dir / 'train'} and {data_dir / 'test'}."
        )
    test_labels = labels_from_paths(test_paths)
    if np.unique(test_labels).size != 2:
        raise ValueError("The labeled test set must contain normal and anomaly clips.")

    print(f"Extracting features from {len(train_paths)} normal training clips...")
    train_features = load_features(train_paths)[:, None, :, :]
    print(f"Extracting features from {len(test_paths)} labeled evaluation clips...")
    test_features = load_features(test_paths)[:, None, :, :]

    mean = train_features.mean(axis=(0, 1, 3))
    std = train_features.std(axis=(0, 1, 3))
    std = np.maximum(std, 1e-4)
    train_features = _normalize(train_features, mean, std)
    test_features = _normalize(test_features, mean, std)

    indices = np.arange(len(test_paths))
    strata = stratification_labels(test_paths, test_labels)
    calibration_indices, evaluation_indices = train_test_split(
        indices,
        test_size=args.evaluation_fraction,
        random_state=args.seed,
        stratify=strata,
    )

    classifier_train_indices, threshold_indices = train_test_split(
        calibration_indices,
        test_size=0.35,
        random_state=args.seed,
        stratify=strata[calibration_indices],
    )
    device = _get_device(args.device)
    architecture = args.architecture
    base_architecture = (
        "large" if architecture in ("large", "large_pruned") else "small"
    )
    prune_keep_fraction = (
        args.prune_keep_fraction if architecture == "large_pruned" else None
    )
    normal_features = (
        _prepare_model_features(train_features, base_architecture)
        if architecture != "small_calibration_only"
        else np.empty((0, N_MELS * 10), dtype=np.float32)
    )
    test_model_features = _prepare_model_features(test_features, base_architecture)
    fit_features = test_model_features[classifier_train_indices]
    fit_labels = test_labels[classifier_train_indices]

    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(
        f"Training {architecture} classifier on {device} ({device_name}) with "
        f"{len(normal_features)} normal clips and "
        f"{len(classifier_train_indices)} labeled calibration clips..."
    )
    model = _fit_classifier(
        fit_features,
        fit_labels,
        normal_features,
        base_architecture,
        device,
        args.seed,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        max(5, args.early_stopping_patience),
        prune_keep_fraction,
    )
    test_features = test_model_features
    hidden_dims = model.hidden_dims
    threshold_scores = _probabilities(
        model, test_features[threshold_indices], device
    )
    threshold, domain_thresholds, calibration_false_positives = _calibrate_thresholds(
        [test_paths[index] for index in threshold_indices],
        test_labels[threshold_indices],
        threshold_scores,
    )
    threshold_validation_thresholds = np.array(
        [
            domain_thresholds.get(_domain(test_paths[index]), threshold)
            for index in threshold_indices
        ]
    )
    evaluation_scores = _probabilities(
        model, test_features[evaluation_indices], device
    )
    evaluation_labels = test_labels[evaluation_indices]
    evaluation_thresholds = np.array(
        [
            domain_thresholds.get(_domain(test_paths[index]), threshold)
            for index in evaluation_indices
        ]
    )
    evaluation_predictions = (evaluation_scores >= evaluation_thresholds).astype(
        np.int64
    )
    threshold_predictions = (
        threshold_scores >= threshold_validation_thresholds
    ).astype(np.int64)
    report = {
        "model": "supervised deep log-mel statistics classifier",
        "architecture": architecture,
        "hidden_dims": list(hidden_dims),
        "threshold": threshold,
        "thresholds_by_domain": domain_thresholds,
        "threshold_optimization_metric": "maximum F1 on threshold-validation clips per domain",
        "threshold_calibrated_on": (
            "labeled threshold-validation subset of the calibration split"
        ),
        "evaluation": _metrics(evaluation_labels, evaluation_predictions),
        "threshold_validation": _metrics(
            test_labels[threshold_indices], threshold_predictions
        ),
        "normalization_normal_clips": len(train_paths),
        "normal_clips_in_classifier_training": len(normal_features),
        "labeled_calibration_training_clips": len(classifier_train_indices),
        "test_clips": len(test_paths),
        "evaluation_fraction": args.evaluation_fraction,
        "seed": args.seed,
        "device": str(device),
    }
    report["calibration_false_positives_at_threshold"] = (
        calibration_false_positives
    )

    checkpoint_path = output_dir / "bearing_classifier.pt"
    torch.save(
        {
            "architecture": architecture,
            "hidden_dims": hidden_dims,
            "state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
            "mean": torch.from_numpy(mean),
            "std": torch.from_numpy(std),
            "threshold": threshold,
            "thresholds_by_domain": domain_thresholds,
            "sample_rate": SAMPLE_RATE,
            "n_mels": N_MELS,
            "n_frames": N_FRAMES,
        },
        checkpoint_path,
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    all_scores = np.empty(len(test_paths), dtype=np.float64)
    all_scores[threshold_indices] = threshold_scores
    all_scores[classifier_train_indices] = _probabilities(
        model, test_features[classifier_train_indices], device
    )
    all_scores[evaluation_indices] = evaluation_scores
    split_by_index = {
        int(index): "classifier_training" for index in classifier_train_indices
    }
    split_by_index.update(
        {int(index): "threshold_validation" for index in threshold_indices}
    )
    split_by_index.update({int(index): "evaluation" for index in evaluation_indices})
    review_indices = np.concatenate([threshold_indices, evaluation_indices])
    review_scores = np.concatenate([threshold_scores, evaluation_scores])
    review_splits = np.concatenate(
        [
            np.full(len(threshold_indices), "threshold_validation"),
            np.full(len(evaluation_indices), "evaluation"),
        ]
    )
    review_rows = []
    for index, score, split in zip(review_indices, review_scores, review_splits):
        if test_labels[index] != 0:
            continue
        path_threshold = domain_thresholds.get(_domain(test_paths[index]), threshold)
        review_rows.append(
            (
                float(score - path_threshold),
                test_paths[index].name,
                "source"
                if "_source_" in test_paths[index].name
                else "target",
                split,
                float(score),
                path_threshold,
                int(score >= path_threshold),
            )
        )
    review_rows.sort(reverse=True)
    with (output_dir / "false_positive_review.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "file",
                "domain",
                "split",
                "anomaly_probability",
                "threshold",
                "margin_over_threshold",
                "false_positive",
            ]
        )
        for (
            margin,
            filename,
            domain,
            split,
            score,
            path_threshold,
            is_false_positive,
        ) in review_rows:
            writer.writerow(
                [
                    filename,
                    domain,
                    split,
                    f"{score:.8f}",
                    f"{path_threshold:.8f}",
                    f"{margin:.8f}",
                    is_false_positive,
                ]
            )
    with (output_dir / "scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "file",
                "label",
                "split",
                "anomaly_probability",
                "prediction",
                "threshold",
            ]
        )
        for index, path in enumerate(test_paths):
            path_threshold = domain_thresholds.get(_domain(path), threshold)
            writer.writerow(
                [
                    path.name,
                    "anomaly" if test_labels[index] else "normal",
                    split_by_index[index],
                    f"{all_scores[index]:.8f}",
                    int(all_scores[index] >= path_threshold),
                    f"{path_threshold:.8f}",
                ]
            )

    print(f"\nEvaluation (held-out labeled clips): {json.dumps(report['evaluation'])}")
    print(f"Calibration false positives: {calibration_false_positives}")
    print(f"Threshold: {threshold:.8f}")
    print(f"Saved model: {checkpoint_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved normal-clip review: {output_dir / 'false_positive_review.csv'}")
    return report


def _aggregate_fold_metrics(folds: list[dict[str, object]]) -> dict[str, object]:
    sample_count = sum(int(fold["metrics"]["samples"]) for fold in folds)
    anomaly_count = sum(int(fold["metrics"]["anomalies"]) for fold in folds)
    false_positives = sum(int(fold["metrics"]["false_positives"]) for fold in folds)
    true_positives = sum(int(fold["metrics"]["true_positives"]) for fold in folds)
    normal_count = sample_count - anomaly_count
    false_negatives = anomaly_count - true_positives
    f1_denominator = 2 * true_positives + false_positives + false_negatives
    return {
        "folds": len(folds),
        "mean_fold_f1": float(
            np.mean([float(fold["metrics"]["f1"]) for fold in folds])
        ),
        "pooled_f1": (
            2 * true_positives / f1_denominator if f1_denominator else 0.0
        ),
        "mean_fold_recall": float(
            np.mean([float(fold["metrics"]["recall"]) for fold in folds])
        ),
        "pooled_recall": true_positives / anomaly_count if anomaly_count else 0.0,
        "false_positives": false_positives,
        "false_positive_rate": false_positives / normal_count if normal_count else 0.0,
        "normal_clips": normal_count,
        "anomaly_clips": anomaly_count,
        "fold_metrics": folds,
    }


def compare_models(args: argparse.Namespace) -> None:
    _set_seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir
    normal_paths = sorted((data_dir / "train").glob("*.wav"))
    labeled_paths = sorted((data_dir / "test").glob("*.wav"))
    if not normal_paths or not labeled_paths:
        raise FileNotFoundError(
            f"Expected WAV files in {data_dir / 'train'} and {data_dir / 'test'}."
        )
    labels = labels_from_paths(labeled_paths)
    strata = stratification_labels(labeled_paths, labels)
    raw_normal = load_features(normal_paths)[:, None, :, :]
    raw_labeled = load_features(labeled_paths)[:, None, :, :]
    mean = raw_normal.mean(axis=(0, 1, 3))
    std = np.maximum(raw_normal.std(axis=(0, 1, 3)), 1e-4)
    normalized_normal = _normalize(raw_normal, mean, std)
    normalized_labeled = _normalize(raw_labeled, mean, std)

    calibration_indices, final_evaluation_indices = train_test_split(
        np.arange(len(labeled_paths)),
        test_size=args.evaluation_fraction,
        random_state=args.seed,
        stratify=strata,
    )
    folds = min(args.folds, min(np.unique(strata[calibration_indices], return_counts=True)[1]))
    if folds < 2:
        raise ValueError("Not enough labeled clips per domain/class for model comparison.")
    splitter = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=args.seed
    )
    device = _get_device(args.device)
    features = _summarize_spectrograms(normalized_labeled)
    normal_features = _summarize_spectrograms(normalized_normal)
    architecture_specs = {
        "small_calibration_only": (
            "small",
            np.empty((0, N_MELS * 10), dtype=np.float32),
        ),
        "small": ("small", normal_features),
        "large": ("large", normal_features),
    }
    fold_results: dict[str, list[dict[str, object]]] = {
        name: [] for name in architecture_specs
    }
    fp_rows: list[tuple[str, int, str, str, float, float]] = []

    for fold_number, (fold_train_relative, fold_eval_relative) in enumerate(
        splitter.split(calibration_indices, strata[calibration_indices]), start=1
    ):
        fold_train_indices = calibration_indices[fold_train_relative]
        fold_eval_indices = calibration_indices[fold_eval_relative]
        classifier_indices, threshold_indices = train_test_split(
            fold_train_indices,
            test_size=args.threshold_fraction,
            random_state=args.seed + fold_number,
            stratify=strata[fold_train_indices],
        )
        fold_labels = labels[fold_eval_indices]
        print(
            f"Comparison fold {fold_number}/{folds}: "
            f"{len(classifier_indices)} fit, {len(threshold_indices)} threshold, "
            f"{len(fold_eval_indices)} validation clips"
        )
        for candidate_name, (architecture, candidate_normal_features) in architecture_specs.items():
            model = _fit_classifier(
                features[classifier_indices],
                labels[classifier_indices],
                candidate_normal_features,
                architecture,
                device,
                args.seed + fold_number,
                args.epochs,
                args.batch_size,
                args.learning_rate,
                max(5, args.early_stopping_patience),
            )
            calibration_scores = _probabilities(
                model, features[threshold_indices], device
            )
            threshold, domain_thresholds, calibration_fp_count = _calibrate_thresholds(
                [labeled_paths[index] for index in threshold_indices],
                labels[threshold_indices],
                calibration_scores,
            )
            eval_scores = _probabilities(model, features[fold_eval_indices], device)
            eval_thresholds = np.array(
                [
                    domain_thresholds.get(_domain(labeled_paths[index]), threshold)
                    for index in fold_eval_indices
                ]
            )
            eval_predictions = (eval_scores >= eval_thresholds).astype(np.int64)
            metrics = _metrics(fold_labels, eval_predictions)
            fold_results[candidate_name].append(
                {
                    "fold": fold_number,
                    "threshold": threshold,
                    "thresholds_by_domain": domain_thresholds,
                    "calibration_false_positives": calibration_fp_count,
                    "calibration_normal_count": int(
                        np.count_nonzero(labels[threshold_indices] == 0)
                    ),
                    "metrics": metrics,
                }
            )
            for index, score, prediction in zip(
                fold_eval_indices, eval_scores, eval_predictions
            ):
                if labels[index] == 0:
                    fp_rows.append(
                        (
                            candidate_name,
                            fold_number,
                            labeled_paths[index].name,
                            "source"
                            if "_source_" in labeled_paths[index].name
                            else "target",
                            float(score),
                            domain_thresholds.get(
                                _domain(labeled_paths[index]), threshold
                            ),
                        )
                    )
            print(
                f"  {candidate_name}: F1={metrics['f1']:.3f}, "
                f"FP={metrics['false_positives']}, "
                f"calibration FP={calibration_fp_count}"
            )

    summaries = {
        name: _aggregate_fold_metrics(results)
        for name, results in fold_results.items()
    }
    small_candidates = ("small_calibration_only", "small")
    baseline_name = max(
        small_candidates,
        key=lambda name: (
            float(summaries[name]["pooled_f1"]),
            -int(summaries[name]["false_positives"]),
        ),
    )
    small_folds = fold_results[baseline_name]
    large_folds = fold_results["large"]
    large_f1_wins = sum(
        float(large["metrics"]["f1"]) > float(small["metrics"]["f1"])
        for small, large in zip(small_folds, large_folds)
    )
    required_wins = int(np.ceil(0.8 * folds))
    large_consistent = (
        large_f1_wins >= required_wins
        and float(summaries["large"]["pooled_f1"])
        > float(summaries[baseline_name]["pooled_f1"])
    )

    pruned_wins = False
    pruned_results: list[dict[str, object]] = []
    if large_consistent:
        print("Large MLP improved consistently; evaluating structured-pruned variant.")
        for fold_number, (fold_train_relative, fold_eval_relative) in enumerate(
            splitter.split(calibration_indices, strata[calibration_indices]), start=1
        ):
            fold_train_indices = calibration_indices[fold_train_relative]
            fold_eval_indices = calibration_indices[fold_eval_relative]
            classifier_indices, threshold_indices = train_test_split(
                fold_train_indices,
                test_size=args.threshold_fraction,
                random_state=args.seed + fold_number,
                stratify=strata[fold_train_indices],
            )
            model = _fit_classifier(
                features[classifier_indices],
                labels[classifier_indices],
                normal_features,
                "large",
                device,
                args.seed + fold_number,
                args.epochs,
                args.batch_size,
                args.learning_rate,
                max(5, args.early_stopping_patience),
                args.prune_keep_fraction,
            )
            calibration_scores = _probabilities(
                model, features[threshold_indices], device
            )
            threshold, domain_thresholds, calibration_fp_count = _calibrate_thresholds(
                [labeled_paths[index] for index in threshold_indices],
                labels[threshold_indices],
                calibration_scores,
            )
            eval_scores = _probabilities(model, features[fold_eval_indices], device)
            eval_thresholds = np.array(
                [
                    domain_thresholds.get(_domain(labeled_paths[index]), threshold)
                    for index in fold_eval_indices
                ]
            )
            metrics = _metrics(
                labels[fold_eval_indices],
                (eval_scores >= eval_thresholds).astype(np.int64),
            )
            pruned_results.append(
                {
                    "fold": fold_number,
                    "threshold": threshold,
                    "thresholds_by_domain": domain_thresholds,
                    "calibration_false_positives": calibration_fp_count,
                    "metrics": metrics,
                }
            )
            for index, score in zip(fold_eval_indices, eval_scores):
                if labels[index] == 0:
                    fp_rows.append(
                        (
                            "large_pruned",
                            fold_number,
                            labeled_paths[index].name,
                            "source"
                            if "_source_" in labeled_paths[index].name
                            else "target",
                            float(score),
                            domain_thresholds.get(
                                _domain(labeled_paths[index]), threshold
                            ),
                        )
                    )
            print(
                f"  large_pruned: F1={metrics['f1']:.3f}, "
                f"FP={metrics['false_positives']}"
            )
        summaries["large_pruned"] = _aggregate_fold_metrics(pruned_results)
        pruned_f1_wins = sum(
            float(pruned["metrics"]["f1"]) > float(small["metrics"]["f1"])
            for small, pruned in zip(small_folds, pruned_results)
        )
        pruned_wins = (
            pruned_f1_wins >= required_wins
            and float(summaries["large_pruned"]["pooled_f1"])
            > float(summaries[baseline_name]["pooled_f1"])
        )

    selected_architecture = (
        "large_pruned"
        if large_consistent and pruned_wins
        else "large"
        if large_consistent
        else baseline_name
    )
    report = {
        "comparison_method": (
            "stratified cross-validation within the calibration partition; "
            "thresholds calibrated independently per fold"
        ),
        "folds": folds,
        "threshold_optimization_metric": "F1, calibrated on each fold's threshold subset",
        "model_selection_metric": "pooled cross-validated F1",
        "large_model_f1_wins": large_f1_wins,
        "large_model_required_f1_wins": required_wins,
        "large_model_consistently_better": large_consistent,
        "best_small_training_strategy": baseline_name,
        "pruned_model_f1_wins": (
            pruned_f1_wins if large_consistent else None
        ),
        "pruning_improvement_survives": pruned_wins if large_consistent else None,
        "selected_architecture": selected_architecture,
        "cv_results": summaries,
        "final_evaluation_reserved": len(final_evaluation_indices),
    }
    with (output_dir / "cross_validation_false_positives.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "model",
                "fold",
                "file",
                "domain",
                "anomaly_probability",
                "calibrated_threshold",
                "false_positive",
            ]
        )
        for model_name, fold_number, filename, domain, score, threshold in fp_rows:
            writer.writerow(
                [
                    model_name,
                    fold_number,
                    filename,
                    domain,
                    f"{score:.8f}",
                    f"{threshold:.8f}",
                    int(score >= threshold),
                ]
            )
    comparison_path = output_dir / "model_comparison.json"
    comparison_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selected architecture: {selected_architecture}")
    print(f"Saved comparison: {comparison_path}")

    final_args = argparse.Namespace(**vars(args))
    final_args.architecture = selected_architecture
    final_report = train(final_args)
    report["final_held_out_evaluation"] = final_report["evaluation"]
    comparison_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def predict(args: argparse.Namespace) -> None:
    device = _get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("architecture", "small")
    hidden_dims = tuple(checkpoint.get("hidden_dims", ARCHITECTURES["small"]))
    base_architecture = (
        "large"
        if architecture in ("large", "large_pruned")
        else "small"
    )
    model = FeatureClassifier(hidden_dims).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    features = load_features(args.files)[:, None, :, :]
    mean = checkpoint["mean"].numpy()
    std = checkpoint["std"].numpy()
    normalized = _prepare_model_features(_normalize(features, mean, std), base_architecture)
    scores = _probabilities(model, normalized, device)
    threshold = float(checkpoint["threshold"])
    domain_thresholds = checkpoint.get("thresholds_by_domain", {})
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Scoring on {device} ({device_name})")
    for path, score in zip(args.files, scores):
        path_threshold = float(domain_thresholds.get(_domain(path), threshold))
        print(
            json.dumps(
                {
                    "file": str(path),
                    "anomaly_probability": float(score),
                    "threshold": path_threshold,
                    "prediction": "anomaly" if score >= path_threshold else "normal",
                }
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and run a local bearing anomaly detector."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train and evaluate the model")
    train_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "bearing",
        help="directory containing train/ and test/ WAV files",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    train_parser.add_argument("--epochs", type=int, default=40)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--early-stopping-patience", type=int, default=7)
    train_parser.add_argument("--evaluation-fraction", type=float, default=0.3)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument(
        "--architecture",
        choices=("small", "small_calibration_only", "large", "large_pruned"),
        default="small",
        help="classifier size; compare command selects this automatically",
    )
    train_parser.add_argument("--prune-keep-fraction", type=float, default=0.7)
    train_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="compute device (auto uses CUDA when available)",
    )
    train_parser.set_defaults(handler=train)

    compare_parser = commands.add_parser(
        "compare", help="compare model training strategies by cross-validated F1"
    )
    compare_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "bearing",
    )
    compare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    compare_parser.add_argument("--epochs", type=int, default=40)
    compare_parser.add_argument("--batch-size", type=int, default=32)
    compare_parser.add_argument("--learning-rate", type=float, default=1e-3)
    compare_parser.add_argument("--early-stopping-patience", type=int, default=7)
    compare_parser.add_argument("--evaluation-fraction", type=float, default=0.3)
    compare_parser.add_argument("--seed", type=int, default=42)
    compare_parser.add_argument("--folds", type=int, default=5)
    compare_parser.add_argument("--threshold-fraction", type=float, default=0.25)
    compare_parser.add_argument("--prune-keep-fraction", type=float, default=0.7)
    compare_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    compare_parser.set_defaults(handler=compare_models, architecture="small")

    predict_parser = commands.add_parser("predict", help="score one or more WAV files")
    predict_parser.add_argument(
        "--checkpoint", type=Path, required=True, help="trained .pt model file"
    )
    predict_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="compute device (auto uses CUDA when available)",
    )
    predict_parser.add_argument("files", type=Path, nargs="+")
    predict_parser.set_defaults(handler=predict)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command in ("train", "compare") and not 0.1 <= args.evaluation_fraction <= 0.8:
        raise ValueError("--evaluation-fraction must be between 0.1 and 0.8.")
    if args.command == "compare" and args.folds < 2:
        raise ValueError("--folds must be at least 2.")
    if args.command == "compare" and not 0.1 <= args.threshold_fraction <= 0.5:
        raise ValueError("--threshold-fraction must be between 0.1 and 0.5.")
    if args.command in ("train", "compare") and not 0.1 <= args.prune_keep_fraction < 1:
        raise ValueError("--prune-keep-fraction must be in [0.1, 1).")
    args.handler(args)


if __name__ == "__main__":
    main()
