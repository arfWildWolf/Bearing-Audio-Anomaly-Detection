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
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    fbeta_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
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


def _extract_mel_feature(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    if waveform.size == 0:
        raise ValueError("Audio channel is empty.")
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


def extract_feature(path: Path) -> np.ndarray:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return _extract_mel_feature(waveform.mean(axis=1), sample_rate)


def extract_dual_channel_features(path: Path) -> np.ndarray:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if waveform.shape[1] != 2:
        raise ValueError(
            f"Dual-channel fusion requires exactly 2 channels (near ch01, far ch02); "
            f"{path} has {waveform.shape[1]}."
        )
    return np.stack(
        [
            _extract_mel_feature(waveform[:, channel], sample_rate)
            for channel in range(2)
        ]
    )


def load_features(paths: list[Path]) -> np.ndarray:
    if not paths:
        raise ValueError("No WAV files were found.")
    return np.stack([extract_feature(path) for path in paths])


def load_dual_channel_features(paths: list[Path]) -> np.ndarray:
    if not paths:
        raise ValueError("No WAV files were found.")
    return np.stack([extract_dual_channel_features(path) for path in paths])


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
    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (256, 64),
        input_dim: int = N_MELS * 10,
    ) -> None:
        super().__init__()
        self.hidden_dims = hidden_dims
        self.input_dim = input_dim
        layers: list[nn.Module] = []
        layer_input_dim = input_dim
        for index, hidden_dim in enumerate(hidden_dims):
            layers.append(
                nn.Sequential(
                    nn.Linear(layer_input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.3 if index == 0 else 0.2),
                )
            )
            layer_input_dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(layer_input_dim, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            values = layer(values)
        return self.classifier(values).squeeze(1)


class RbfSVMScorer(nn.Module):
    def __init__(
        self,
        support_vectors: np.ndarray,
        dual_coefficients: np.ndarray,
        intercept: float,
        gamma: float,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "support_vectors", torch.from_numpy(support_vectors.astype(np.float32))
        )
        self.register_buffer(
            "dual_coefficients",
            torch.from_numpy(dual_coefficients.astype(np.float32)),
        )
        self.register_buffer("intercept", torch.tensor([intercept], dtype=torch.float32))
        self.register_buffer("gamma", torch.tensor([gamma], dtype=torch.float32))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values_squared = values.square().sum(dim=1, keepdim=True)
        support_squared = self.support_vectors.square().sum(dim=1).unsqueeze(0)
        distances_squared = (
            values_squared
            + support_squared
            - 2 * values @ self.support_vectors.T
        ).clamp_min_(0)
        kernels = torch.exp(-self.gamma * distances_squared)
        return (kernels @ self.dual_coefficients.unsqueeze(1)).squeeze(1) + self.intercept


class AutoencoderFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(values)
        return torch.cat(
            [
                latent.mean(dim=(2, 3)),
                latent.std(dim=(2, 3), unbiased=False),
            ],
            dim=1,
        )


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
    pruned = FeatureClassifier(hidden_dims, model.input_dim).to(
        next(model.parameters()).device
    )
    previous_indices = torch.arange(
        model.input_dim, device=next(model.parameters()).device
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
    if architecture not in ARCHITECTURES and architecture != "rbf_svm":
        raise ValueError(f"Unknown model architecture: {architecture}")
    return _summarize_spectrograms(raw)


def _normalize_dual_channel(
    features: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return np.clip(
        (features - mean[None, :, :, None]) / std[None, :, :, None],
        -8.0,
        8.0,
    ).astype(np.float32)


def _summarize_dual_channel(features: np.ndarray) -> np.ndarray:
    if features.ndim != 4 or features.shape[1] != 2:
        raise ValueError(
            "Expected normalized spectrograms shaped (clips, 2, mels, frames)."
        )
    near = _summarize_spectrograms(features[:, 0:1])
    far = _summarize_spectrograms(features[:, 1:2])
    return np.concatenate([near, near - far], axis=1).astype(np.float32)


def _probabilities(
    model: nn.Module, features: np.ndarray, device: torch.device
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


def _fit_knn_distance_reference(
    normal_features: np.ndarray,
    device: torch.device,
    neighbors: int = 10,
    feature_block_dims: tuple[int, ...] | None = None,
) -> dict[str, object]:
    if len(normal_features) < 2:
        raise ValueError("kNN score normalization requires at least two normal clips.")
    mean = normal_features.mean(axis=0)
    std = np.maximum(normal_features.std(axis=0), 1e-6)
    standardized = np.clip((normal_features - mean) / std, -8.0, 8.0)
    block_dims = feature_block_dims or (normal_features.shape[1],)
    if sum(block_dims) != normal_features.shape[1] or any(
        block_dim <= 0 for block_dim in block_dims
    ):
        raise ValueError("Feature block dimensions must partition the input features.")
    block_scales = tuple(
        float(np.sqrt(1.0 / (len(block_dims) * block_dim)))
        for block_dim in block_dims
    )
    offset = 0
    for block_dim, scale in zip(block_dims, block_scales):
        standardized[:, offset : offset + block_dim] *= scale
        offset += block_dim
    reference = torch.from_numpy(standardized.astype(np.float32)).to(device)
    k = min(neighbors, len(reference) - 1)
    leave_one_out_distances: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(reference), 64):
            stop = min(start + 64, len(reference))
            distances = torch.cdist(reference[start:stop], reference)
            row_indices = torch.arange(start, stop, device=device)
            distances[
                torch.arange(stop - start, device=device), row_indices
            ] = float("inf")
            leave_one_out_distances.append(
                distances.topk(k, largest=False).values.mean(dim=1)
            )
    baseline_distance = float(
        torch.cat(leave_one_out_distances).median().item()
    )
    if not np.isfinite(baseline_distance) or baseline_distance <= 0:
        raise ValueError("Normal-reference kNN distance is not finite and positive.")
    return {
        "mean": torch.from_numpy(mean.astype(np.float32)).to(device),
        "std": torch.from_numpy(std.astype(np.float32)).to(device),
        "reference": reference,
        "baseline_distance": baseline_distance,
        "neighbors": k,
        "alpha": 1.0,
        "block_dims": block_dims,
        "block_scales": block_scales,
    }


def _apply_knn_score_normalization(
    scores: np.ndarray,
    features: np.ndarray,
    reference_stats: dict[str, object],
    device: torch.device,
) -> np.ndarray:
    reference = reference_stats["reference"]
    mean = reference_stats["mean"]
    std = reference_stats["std"]
    if not isinstance(reference, torch.Tensor) or not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
        raise TypeError("Invalid kNN normalizer tensors.")
    baseline_distance = float(reference_stats["baseline_distance"])
    neighbors = int(reference_stats["neighbors"])
    alpha = float(reference_stats["alpha"])
    normalized_scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), 32):
            query = torch.from_numpy(features[start : start + 32]).to(device)
            query = ((query - mean) / std).clamp(-8.0, 8.0)
            block_dims = tuple(
                reference_stats.get("block_dims", (query.shape[1],))
            )
            block_scales = tuple(
                reference_stats.get("block_scales", (1.0,) * len(block_dims))
            )
            if len(block_dims) != len(block_scales) or sum(block_dims) != query.shape[1]:
                raise ValueError("Invalid kNN feature-block calibration data.")
            offset = 0
            for block_dim, scale in zip(block_dims, block_scales):
                query[:, offset : offset + block_dim] *= scale
                offset += block_dim
            distances = torch.cdist(query, reference)
            mean_neighbor_distance = distances.topk(
                neighbors, largest=False
            ).values.mean(dim=1)
            distance_ratio = (mean_neighbor_distance / baseline_distance).clamp_min(1e-4)
            raw_scores = torch.from_numpy(
                scores[start : start + 32].astype(np.float32)
            ).to(device).clamp(1e-6, 1.0 - 1e-6)
            score_logits = torch.logit(raw_scores)
            adjusted_logits = score_logits + alpha * torch.log(distance_ratio)
            normalized_scores.append(torch.sigmoid(adjusted_logits).cpu().numpy())
    return np.concatenate(normalized_scores)


def _build_autoencoder_extractor(
    state_dict: dict[str, torch.Tensor], device: torch.device
) -> AutoencoderFeatureExtractor:
    extractor = AutoencoderFeatureExtractor()
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    extractor.encoder.load_state_dict(encoder_state, strict=True)
    return extractor.to(device).eval()


def _load_autoencoder_extractor(
    checkpoint_path: Path, device: torch.device
) -> tuple[
    AutoencoderFeatureExtractor,
    np.ndarray,
    np.ndarray,
    dict[str, torch.Tensor],
]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Autoencoder feature checkpoint not found: {checkpoint_path}"
        )
    numpy_core = getattr(np, "_core", None)
    if numpy_core is None:
        numpy_core = np.core
    safe_numpy_types = [
        numpy_core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.float32)),
    ]
    with torch.serialization.safe_globals(safe_numpy_types):
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Autoencoder checkpoint does not contain a state_dict.")
    extractor = _build_autoencoder_extractor(state_dict, device)
    return (
        extractor,
        np.asarray(checkpoint["mean"], dtype=np.float32),
        np.asarray(checkpoint["std"], dtype=np.float32),
        {
            key: value.detach().cpu()
            for key, value in state_dict.items()
            if key.startswith("encoder.")
        },
    )


def _autoencoder_features(
    raw_features: np.ndarray,
    extractor: AutoencoderFeatureExtractor,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    normalized = _normalize(raw_features, mean, std)
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(normalized), 32):
            values = torch.from_numpy(normalized[start : start + 32]).to(device)
            batches.append(extractor(values).cpu().numpy())
    return np.concatenate(batches).astype(np.float32)


def _stack_knn_features(
    raw_features: np.ndarray, autoencoder_features: np.ndarray
) -> np.ndarray:
    if len(raw_features) != len(autoencoder_features):
        raise ValueError("Raw and autoencoder feature batches must be aligned.")
    return np.concatenate([raw_features, autoencoder_features], axis=1)


def _fit_rbf_svm(
    labeled_features: np.ndarray,
    labels: np.ndarray,
    normal_features: np.ndarray,
    device: torch.device,
) -> RbfSVMScorer:
    features = np.concatenate([normal_features, labeled_features], axis=0)
    combined_labels = np.concatenate(
        [np.zeros(len(normal_features), dtype=np.int64), labels]
    )
    estimator = SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        class_weight="balanced",
    )
    estimator.fit(features, combined_labels)
    feature_variance = float(features.var())
    gamma = (
        1.0 / (features.shape[1] * feature_variance)
        if feature_variance > 0
        else 1.0
    )
    return RbfSVMScorer(
        estimator.support_vectors_,
        estimator.dual_coef_[0],
        float(estimator.intercept_[0]),
        gamma,
    ).to(device)


def _fit_normal_rank_calibrator(
    paths: list[Path],
    labels: np.ndarray,
    scores: np.ndarray,
    maximum_false_positive_rate: float,
) -> dict[str, object]:
    if not 0 < maximum_false_positive_rate < 1:
        raise ValueError("Maximum false-positive rate must be between 0 and 1.")
    normal_mask = labels == 0
    if not normal_mask.any():
        raise ValueError("Normal-rank calibration requires normal clips.")
    overall_reference = np.sort(scores[normal_mask].astype(np.float64))
    overall_raw_threshold = _maximum_fpr_threshold(
        overall_reference, maximum_false_positive_rate
    )
    references_by_domain: dict[str, np.ndarray] = {}
    raw_thresholds_by_domain: dict[str, float] = {}
    for domain in ("source", "target"):
        domain_mask = normal_mask & np.array(
            [_domain(path) == domain for path in paths]
        )
        if not domain_mask.any():
            continue
        reference = np.sort(scores[domain_mask].astype(np.float64))
        references_by_domain[domain] = reference
        raw_thresholds_by_domain[domain] = _maximum_fpr_threshold(
            reference, maximum_false_positive_rate
        )
    rank_threshold = _normal_rank_scores(
        np.array([overall_raw_threshold]), [Path("unknown")], overall_reference,
        references_by_domain
    )[0]
    rank_thresholds_by_domain = {
        domain: float(
            _normal_rank_scores(
                np.array([raw_threshold]),
                [Path(f"_{domain}_")],
                overall_reference,
                references_by_domain,
            )[0]
        )
        for domain, raw_threshold in raw_thresholds_by_domain.items()
    }
    return {
        "overall_reference": overall_reference,
        "references_by_domain": references_by_domain,
        "overall_raw_threshold": overall_raw_threshold,
        "raw_thresholds_by_domain": raw_thresholds_by_domain,
        "overall_rank_threshold": float(rank_threshold),
        "rank_thresholds_by_domain": rank_thresholds_by_domain,
        "maximum_false_positive_rate": maximum_false_positive_rate,
        "normal_count": int(np.count_nonzero(normal_mask)),
        "allowed_false_positives": max(
            0,
            int(np.ceil(maximum_false_positive_rate * np.count_nonzero(normal_mask))) - 1,
        ),
    }


def _normal_rank_scores(
    scores: np.ndarray,
    paths: list[Path],
    overall_reference: np.ndarray,
    references_by_domain: dict[str, np.ndarray],
) -> np.ndarray:
    ranked = np.empty(len(scores), dtype=np.float64)
    for index, (score, path) in enumerate(zip(scores, paths)):
        domain = _domain(path)
        reference = references_by_domain.get(domain, overall_reference)
        ranked[index] = np.searchsorted(reference, score, side="left") / len(reference)
    return ranked


def _maximum_fpr_threshold(
    sorted_normal_scores: np.ndarray, maximum_false_positive_rate: float
) -> float:
    if sorted_normal_scores.size == 0:
        raise ValueError("Threshold selection requires normal scores.")
    allowed_false_positives = max(
        0,
        int(np.ceil(maximum_false_positive_rate * len(sorted_normal_scores))) - 1,
    )
    descending = np.sort(sorted_normal_scores)[::-1]
    boundary_index = min(allowed_false_positives, len(descending) - 1)
    return float(np.nextafter(descending[boundary_index], np.inf))


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


def _metrics(
    labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray
) -> dict[str, object]:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negatives, false_positives = matrix[0]
    false_negatives, true_positives = matrix[1]
    precision = float(precision_score(labels, predictions, zero_division=0))
    recall = float(recall_score(labels, predictions, zero_division=0))
    specificity = float(
        true_negatives / (true_negatives + false_positives)
        if true_negatives + false_positives
        else 0.0
    )
    negative_predictive_value = float(
        true_negatives / (true_negatives + false_negatives)
        if true_negatives + false_negatives
        else 0.0
    )
    precision_recall_sum = precision + recall
    harmonic_mean = (
        2 * precision * recall / precision_recall_sum
        if precision_recall_sum
        else 0.0
    )
    has_both_classes = np.unique(labels).size == 2
    return {
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "f2": float(fbeta_score(labels, predictions, beta=2, zero_division=0)),
        "harmonic_mean_precision_recall": float(harmonic_mean),
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "negative_predictive_value": negative_predictive_value,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "geometric_mean_sensitivity_specificity": float(
            np.sqrt(recall * specificity)
        ),
        "matthews_correlation_coefficient": float(
            matthews_corrcoef(labels, predictions)
        ),
        "roc_auc": (
            float(roc_auc_score(labels, scores)) if has_both_classes else None
        ),
        "partial_roc_auc_fpr_0_1": (
            float(roc_auc_score(labels, scores, max_fpr=0.1))
            if has_both_classes
            else None
        ),
        "average_precision": (
            float(average_precision_score(labels, scores))
            if has_both_classes
            else None
        ),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "true_positives": int(matrix[1, 1]),
        "true_negatives": int(true_negatives),
        "false_positive_rate": float(1.0 - specificity),
        "false_negative_rate": float(
            false_negatives / (false_negatives + true_positives)
            if false_negatives + true_positives
            else 0.0
        ),
        "fowlkes_mallows_index": float(
            np.sqrt(precision * recall) if precision * recall else 0.0
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
    input_dim: int = N_MELS * 10,
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

    model = FeatureClassifier(ARCHITECTURES[architecture], input_dim).to(device)
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
    architecture = args.architecture
    use_dual_channel = architecture == "small_dual_channel_normal_rank"

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
    raw_train_features = (
        load_dual_channel_features(train_paths)
        if use_dual_channel
        else load_features(train_paths)[:, None, :, :]
    )
    print(f"Extracting features from {len(test_paths)} labeled evaluation clips...")
    raw_test_features = (
        load_dual_channel_features(test_paths)
        if use_dual_channel
        else load_features(test_paths)[:, None, :, :]
    )

    if use_dual_channel:
        mean = raw_train_features.mean(axis=(0, 3))
        std = raw_train_features.std(axis=(0, 3))
        normalize = _normalize_dual_channel
        summarize = _summarize_dual_channel
        model_input_dim = 2 * N_MELS * 10
    else:
        mean = raw_train_features.mean(axis=(0, 1, 3))
        std = raw_train_features.std(axis=(0, 1, 3))
        normalize = _normalize
        summarize = lambda features: _prepare_model_features(features, "small")
        model_input_dim = N_MELS * 10
    std = np.maximum(std, 1e-4)
    train_features = normalize(raw_train_features, mean, std)
    test_features = normalize(raw_test_features, mean, std)

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
    use_normal_rank = architecture.endswith("_normal_rank")
    architecture_without_rank = (
        architecture.removesuffix("_normal_rank")
        if use_normal_rank
        else architecture
    )
    use_autoencoder_features = architecture_without_rank.endswith("_autoencoder")
    architecture_without_autoencoder = (
        architecture_without_rank.removesuffix("_autoencoder")
        if use_autoencoder_features
        else architecture_without_rank
    )
    use_knn_normalization = architecture_without_autoencoder.endswith("_knn")
    model_architecture = architecture_without_autoencoder.removesuffix("_knn")
    base_architecture = model_architecture
    base_architecture = (
        "large" if base_architecture in ("large", "large_pruned") else "small"
    )
    prune_keep_fraction = (
        args.prune_keep_fraction if model_architecture == "large_pruned" else None
    )
    normal_features = (
        summarize(train_features)
        if model_architecture != "small_calibration_only"
        else np.empty((0, N_MELS * 10), dtype=np.float32)
    )
    knn_reference_features = summarize(train_features)
    test_model_features = summarize(test_features)
    autoencoder_extractor = None
    autoencoder_state_dict = None
    autoencoder_mean = None
    autoencoder_std = None
    normal_autoencoder_features = None
    test_autoencoder_features = None
    if use_autoencoder_features:
        (
            autoencoder_extractor,
            autoencoder_mean,
            autoencoder_std,
            autoencoder_state_dict,
        ) = _load_autoencoder_extractor(args.autoencoder_checkpoint, device)
        normal_autoencoder_features = _autoencoder_features(
            raw_train_features,
            autoencoder_extractor,
            autoencoder_mean,
            autoencoder_std,
            device,
        )
        test_autoencoder_features = _autoencoder_features(
            raw_test_features,
            autoencoder_extractor,
            autoencoder_mean,
            autoencoder_std,
            device,
        )
        knn_reference_features = _stack_knn_features(
            knn_reference_features, normal_autoencoder_features
        )
        test_knn_features = _stack_knn_features(
            test_model_features, test_autoencoder_features
        )
    else:
        test_knn_features = test_model_features
    fit_features = test_model_features[classifier_train_indices]
    fit_labels = test_labels[classifier_train_indices]

    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    if model_architecture == "rbf_svm":
        print(
            f"Training {architecture} classifier on CPU and scoring on {device} "
            f"({device_name}) with {len(normal_features)} normal clips and "
            f"{len(classifier_train_indices)} labeled calibration clips..."
        )
        model = _fit_rbf_svm(fit_features, fit_labels, normal_features, device)
    else:
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
            model_input_dim,
        )
    test_features = test_model_features
    hidden_dims = model.hidden_dims if isinstance(model, FeatureClassifier) else ()
    knn_reference_stats = (
        _fit_knn_distance_reference(
            knn_reference_features,
            device,
            feature_block_dims=(
                (test_model_features.shape[1], test_autoencoder_features.shape[1])
                if use_autoencoder_features
                else None
            ),
        )
        if use_knn_normalization
        else None
    )
    threshold_scores = _probabilities(
        model, test_features[threshold_indices], device
    )
    evaluation_scores = _probabilities(
        model, test_features[evaluation_indices], device
    )
    classifier_train_scores = _probabilities(
        model, test_features[classifier_train_indices], device
    )
    if knn_reference_stats is not None:
        threshold_scores = _apply_knn_score_normalization(
            threshold_scores,
            test_knn_features[threshold_indices],
            knn_reference_stats,
            device,
        )
        evaluation_scores = _apply_knn_score_normalization(
            evaluation_scores,
            test_knn_features[evaluation_indices],
            knn_reference_stats,
            device,
        )
        classifier_train_scores = _apply_knn_score_normalization(
            classifier_train_scores,
            test_knn_features[classifier_train_indices],
            knn_reference_stats,
            device,
        )
    rank_calibrator = _fit_normal_rank_calibrator(
        [test_paths[index] for index in threshold_indices],
        test_labels[threshold_indices],
        threshold_scores,
        args.maximum_false_positive_rate,
    )
    if use_normal_rank:
        threshold_scores = _normal_rank_scores(
            threshold_scores,
            [test_paths[index] for index in threshold_indices],
            rank_calibrator["overall_reference"],
            rank_calibrator["references_by_domain"],
        )
        evaluation_scores = _normal_rank_scores(
            evaluation_scores,
            [test_paths[index] for index in evaluation_indices],
            rank_calibrator["overall_reference"],
            rank_calibrator["references_by_domain"],
        )
        classifier_train_scores = _normal_rank_scores(
            classifier_train_scores,
            [test_paths[index] for index in classifier_train_indices],
            rank_calibrator["overall_reference"],
            rank_calibrator["references_by_domain"],
        )
        threshold = float(rank_calibrator["overall_rank_threshold"])
        domain_thresholds = rank_calibrator["rank_thresholds_by_domain"]
    else:
        threshold = float(rank_calibrator["overall_raw_threshold"])
        domain_thresholds = rank_calibrator["raw_thresholds_by_domain"]
    threshold_validation_thresholds = np.array(
        [
            domain_thresholds.get(_domain(test_paths[index]), threshold)
            for index in threshold_indices
        ]
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
    calibration_false_positives = int(
        np.count_nonzero(
            (threshold_predictions == 1) & (test_labels[threshold_indices] == 0)
        )
    )
    report = {
        "model": "supervised deep log-mel statistics classifier",
        "architecture": architecture,
        "hidden_dims": list(hidden_dims),
        "feature_transform": (
            "640-d near-channel log-mel summary concatenated with its difference from the far-channel summary"
            if use_dual_channel
            else "640-d mono log-mel summary"
        ),
        "threshold": threshold,
        "thresholds_by_domain": domain_thresholds,
        "threshold_optimization_metric": (
            "maximum recall subject to calibration false-positive-rate < configured cap"
        ),
        "maximum_false_positive_rate": args.maximum_false_positive_rate,
        "score_normalization": (
            "balanced raw/autoencoder kNN distance followed by empirical normal-validation rank"
            if use_normal_rank and use_autoencoder_features
            else "k-nearest-normal odds adjustment followed by empirical normal-validation rank"
            if use_normal_rank and use_knn_normalization
            else "empirical normal-validation rank"
            if use_normal_rank
            else "balanced raw/autoencoder kNN distance adjustment"
            if use_autoencoder_features
            else "k-nearest-normal training-distance odds adjustment"
            if use_knn_normalization
            else "none"
        ),
        "knn_feature_stack": (
            "log-mel statistics + locally saved autoencoder latent mean/std"
            if use_autoencoder_features
            else "log-mel statistics"
        ),
        "threshold_calibrated_on": (
            "labeled threshold-validation subset of the calibration split"
        ),
        "evaluation": _metrics(
            evaluation_labels, evaluation_predictions, evaluation_scores
        ),
        "threshold_validation": _metrics(
            test_labels[threshold_indices], threshold_predictions, threshold_scores
        ),
        "normalization_normal_clips": len(train_paths),
        "normal_clips_in_classifier_training": len(normal_features),
        "labeled_calibration_training_clips": len(classifier_train_indices),
        "test_clips": len(test_paths),
        "evaluation_fraction": args.evaluation_fraction,
        "seed": args.seed,
        "device": str(device),
        "classifier_fit_device": "cpu" if model_architecture == "rbf_svm" else str(device),
        "score_is_calibrated_probability": (
            model_architecture != "rbf_svm"
            and not use_knn_normalization
            and not use_normal_rank
        ),
    }
    report["calibration_false_positives_at_threshold"] = (
        calibration_false_positives
    )

    checkpoint_path = output_dir / "bearing_classifier.pt"
    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    torch.save(
        {
            "architecture": architecture,
            "hidden_dims": hidden_dims,
            "input_dim": model.input_dim if isinstance(model, FeatureClassifier) else None,
            "state_dict": state_dict,
            "mean": torch.from_numpy(mean),
            "std": torch.from_numpy(std),
            "threshold": threshold,
            "thresholds_by_domain": domain_thresholds,
            "normal_rank_calibrator": (
                {
                    "overall_reference": torch.from_numpy(
                        rank_calibrator["overall_reference"].astype(np.float32)
                    ),
                    "references_by_domain": {
                        domain: torch.from_numpy(reference.astype(np.float32))
                        for domain, reference in rank_calibrator[
                            "references_by_domain"
                        ].items()
                    },
                    "overall_rank_threshold": rank_calibrator[
                        "overall_rank_threshold"
                    ],
                    "rank_thresholds_by_domain": rank_calibrator[
                        "rank_thresholds_by_domain"
                    ],
                }
                if use_normal_rank
                else None
            ),
            "autoencoder_feature_extractor": (
                {
                    "state_dict": autoencoder_state_dict,
                    "mean": torch.from_numpy(autoencoder_mean),
                    "std": torch.from_numpy(autoencoder_std),
                }
                if use_autoencoder_features
                else None
            ),
            "knn_normalizer": (
                {
                    "mean": knn_reference_stats["mean"].detach().cpu(),
                    "std": knn_reference_stats["std"].detach().cpu(),
                    "reference": knn_reference_stats["reference"].detach().cpu(),
                    "baseline_distance": knn_reference_stats["baseline_distance"],
                    "neighbors": knn_reference_stats["neighbors"],
                    "alpha": knn_reference_stats["alpha"],
                    "block_dims": knn_reference_stats["block_dims"],
                    "block_scales": knn_reference_stats["block_scales"],
                }
                if knn_reference_stats is not None
                else None
            ),
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
    all_scores[classifier_train_indices] = classifier_train_scores
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
                "anomaly_score",
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
                "anomaly_score",
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
    f2_denominator = 5 * true_positives + 4 * false_negatives + false_positives
    mean_fold_metrics = {
        f"mean_fold_{name}": float(
            np.mean(
                [
                    float(fold["metrics"][name])
                    for fold in folds
                    if fold["metrics"][name] is not None
                ]
            )
        )
        for name in (
            "roc_auc",
            "partial_roc_auc_fpr_0_1",
            "average_precision",
            "accuracy",
            "balanced_accuracy",
            "harmonic_mean_precision_recall",
            "specificity",
            "matthews_correlation_coefficient",
        )
        if any(fold["metrics"][name] is not None for fold in folds)
    }
    return {
        "folds": len(folds),
        "mean_fold_f2": float(
            np.mean([float(fold["metrics"]["f2"]) for fold in folds])
        ),
        "pooled_f2": 5 * true_positives / f2_denominator if f2_denominator else 0.0,
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
        **mean_fold_metrics,
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
    (
        autoencoder_extractor,
        autoencoder_mean,
        autoencoder_std,
        _,
    ) = _load_autoencoder_extractor(args.autoencoder_checkpoint, device)
    normal_autoencoder_features = _autoencoder_features(
        raw_normal,
        autoencoder_extractor,
        autoencoder_mean,
        autoencoder_std,
        device,
    )
    labeled_autoencoder_features = _autoencoder_features(
        raw_labeled,
        autoencoder_extractor,
        autoencoder_mean,
        autoencoder_std,
        device,
    )
    stacked_normal_features = _stack_knn_features(
        normal_features, normal_autoencoder_features
    )
    stacked_labeled_features = _stack_knn_features(
        features, labeled_autoencoder_features
    )
    architecture_specs = {
        "small_calibration_only": (
            "small",
            np.empty((0, N_MELS * 10), dtype=np.float32),
            False,
            False,
        ),
        "small_calibration_only_knn": (
            "small",
            np.empty((0, N_MELS * 10), dtype=np.float32),
            True,
            False,
        ),
        "small": ("small", normal_features, False, False),
        "small_knn": ("small", normal_features, True, False),
        "small_knn_autoencoder": (
            "small",
            normal_features,
            True,
            True,
        ),
        "large": ("large", normal_features, False, False),
        "large_knn": ("large", normal_features, True, False),
        "rbf_svm": ("rbf_svm", normal_features, False, False),
        "rbf_svm_knn": ("rbf_svm", normal_features, True, False),
    }
    fold_results: dict[str, list[dict[str, object]]] = {
        name: [] for name in architecture_specs
    }
    fp_rows: list[tuple[str, int, str, str, float, float]] = []
    knn_reference_stats = _fit_knn_distance_reference(normal_features, device)
    stacked_knn_reference_stats = _fit_knn_distance_reference(
        stacked_normal_features,
        device,
        feature_block_dims=(
            normal_features.shape[1],
            normal_autoencoder_features.shape[1],
        ),
    )

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
        for candidate_name, (
            architecture,
            candidate_normal_features,
            use_knn_normalization,
            use_autoencoder_features,
        ) in architecture_specs.items():
            if architecture.removesuffix("_knn") == "rbf_svm":
                model = _fit_rbf_svm(
                    features[classifier_indices],
                    labels[classifier_indices],
                    candidate_normal_features,
                    device,
                )
            else:
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
            eval_scores = _probabilities(model, features[fold_eval_indices], device)
            if use_knn_normalization:
                knn_features = (
                    stacked_labeled_features
                    if use_autoencoder_features
                    else features
                )
                reference_stats = (
                    stacked_knn_reference_stats
                    if use_autoencoder_features
                    else knn_reference_stats
                )
                calibration_scores = _apply_knn_score_normalization(
                    calibration_scores,
                    knn_features[threshold_indices],
                    reference_stats,
                    device,
                )
                eval_scores = _apply_knn_score_normalization(
                    eval_scores,
                    knn_features[fold_eval_indices],
                    reference_stats,
                    device,
                )
            rank_calibrator = _fit_normal_rank_calibrator(
                [labeled_paths[index] for index in threshold_indices],
                labels[threshold_indices],
                calibration_scores,
                args.maximum_false_positive_rate,
            )
            calibration_scores = _normal_rank_scores(
                calibration_scores,
                [labeled_paths[index] for index in threshold_indices],
                rank_calibrator["overall_reference"],
                rank_calibrator["references_by_domain"],
            )
            eval_scores = _normal_rank_scores(
                eval_scores,
                [labeled_paths[index] for index in fold_eval_indices],
                rank_calibrator["overall_reference"],
                rank_calibrator["references_by_domain"],
            )
            threshold = float(rank_calibrator["overall_rank_threshold"])
            domain_thresholds = rank_calibrator["rank_thresholds_by_domain"]
            eval_thresholds = np.array(
                [
                    domain_thresholds.get(_domain(labeled_paths[index]), threshold)
                    for index in fold_eval_indices
                ]
            )
            eval_predictions = (eval_scores >= eval_thresholds).astype(np.int64)
            calibration_thresholds = np.array(
                [
                    domain_thresholds.get(
                        _domain(labeled_paths[index]), threshold
                    )
                    for index in threshold_indices
                ]
            )
            calibration_fp_count = int(
                np.count_nonzero(
                    (calibration_scores >= calibration_thresholds)
                    & (labels[threshold_indices] == 0)
                )
            )
            metrics = _metrics(fold_labels, eval_predictions, eval_scores)
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
                f"  {candidate_name}: pAUC@0.1="
                f"{metrics['partial_roc_auc_fpr_0_1']:.3f}, "
                f"FP={metrics['false_positives']}, "
                f"threshold-subset FP={calibration_fp_count}"
            )

    summaries = {
        name: _aggregate_fold_metrics(results)
        for name, results in fold_results.items()
    }
    small_candidates = (
        "small_calibration_only",
        "small_calibration_only_knn",
        "small",
        "small_knn",
        "small_knn_autoencoder",
    )
    baseline_name = max(
        small_candidates,
        key=lambda name: (
            float(summaries[name]["mean_fold_f1"]),
            -int(summaries[name]["false_positives"]),
        ),
    )
    small_folds = fold_results[baseline_name]
    required_wins = int(np.ceil(0.8 * folds))
    model_wins: dict[str, int] = {}
    consistent_candidates: list[str] = []
    for candidate_name in ("large", "large_knn", "rbf_svm", "rbf_svm_knn"):
        wins = sum(
            float(candidate["metrics"]["f1"]) > float(small["metrics"]["f1"])
            for small, candidate in zip(small_folds, fold_results[candidate_name])
        )
        model_wins[candidate_name] = wins
        if (
            wins >= required_wins
            and float(summaries[candidate_name]["mean_fold_f1"])
            > float(summaries[baseline_name]["mean_fold_f1"])
        ):
            consistent_candidates.append(candidate_name)
    large_consistent = "large" in consistent_candidates
    svm_consistent = "rbf_svm" in consistent_candidates

    pruned_wins = False
    pruned_results: list[dict[str, object]] = []
    pruned_f1_wins: int | None = None
    if large_consistent:
        print("Large MLP improved consistently; evaluating structured-pruned variant.")
        use_knn_for_pruned = baseline_name.endswith("_knn")
        use_autoencoder_for_pruned = baseline_name.endswith("_autoencoder")
        pruned_knn_reference_features = (
            stacked_normal_features
            if use_autoencoder_for_pruned
            else normal_features
        )
        pruned_knn_reference_stats = (
            stacked_knn_reference_stats
            if use_autoencoder_for_pruned
            else knn_reference_stats
        )
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
            eval_scores = _probabilities(model, features[fold_eval_indices], device)
            if use_knn_for_pruned:
                pruned_knn_features = (
                    stacked_labeled_features
                    if use_autoencoder_for_pruned
                    else features
                )
                calibration_scores = _apply_knn_score_normalization(
                    calibration_scores,
                    pruned_knn_features[threshold_indices],
                    pruned_knn_reference_stats,
                    device,
                )
                eval_scores = _apply_knn_score_normalization(
                    eval_scores,
                    pruned_knn_features[fold_eval_indices],
                    pruned_knn_reference_stats,
                    device,
                )
            rank_calibrator = _fit_normal_rank_calibrator(
                [labeled_paths[index] for index in threshold_indices],
                labels[threshold_indices],
                calibration_scores,
                args.maximum_false_positive_rate,
            )
            calibration_scores = _normal_rank_scores(
                calibration_scores,
                [labeled_paths[index] for index in threshold_indices],
                rank_calibrator["overall_reference"],
                rank_calibrator["references_by_domain"],
            )
            eval_scores = _normal_rank_scores(
                eval_scores,
                [labeled_paths[index] for index in fold_eval_indices],
                rank_calibrator["overall_reference"],
                rank_calibrator["references_by_domain"],
            )
            threshold = float(rank_calibrator["overall_rank_threshold"])
            domain_thresholds = rank_calibrator["rank_thresholds_by_domain"]
            eval_thresholds = np.array(
                [
                    domain_thresholds.get(_domain(labeled_paths[index]), threshold)
                    for index in fold_eval_indices
                ]
            )
            calibration_thresholds = np.array(
                [
                    domain_thresholds.get(
                        _domain(labeled_paths[index]), threshold
                    )
                    for index in threshold_indices
                ]
            )
            calibration_fp_count = int(
                np.count_nonzero(
                    (calibration_scores >= calibration_thresholds)
                    & (labels[threshold_indices] == 0)
                )
            )
            metrics = _metrics(
                labels[fold_eval_indices],
                (eval_scores >= eval_thresholds).astype(np.int64),
                eval_scores,
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
                f"  large_pruned: pAUC@0.1="
                f"{metrics['partial_roc_auc_fpr_0_1']:.3f}, "
                f"FP={metrics['false_positives']}"
            )
        pruned_name = (
            "large_pruned_knn_autoencoder"
            if use_knn_for_pruned and use_autoencoder_for_pruned
            else "large_pruned_knn"
            if use_knn_for_pruned
            else "large_pruned"
        )
        summaries[pruned_name] = _aggregate_fold_metrics(pruned_results)
        pruned_f1_wins = sum(
            float(pruned["metrics"]["f1"]) > float(small["metrics"]["f1"])
            for small, pruned in zip(small_folds, pruned_results)
        )
        pruned_wins = (
            pruned_f1_wins >= required_wins
            and float(summaries[pruned_name]["mean_fold_f1"])
            > float(summaries[baseline_name]["mean_fold_f1"])
        )

    eligible_candidates = [baseline_name]
    if large_consistent:
        eligible_candidates.append(
            pruned_name if pruned_wins else "large"
        )
    if "large_knn" in consistent_candidates:
        eligible_candidates.append("large_knn")
    if svm_consistent:
        eligible_candidates.append("rbf_svm")
    if "rbf_svm_knn" in consistent_candidates:
        eligible_candidates.append("rbf_svm_knn")
    selected_architecture = max(
        eligible_candidates,
        key=lambda name: (
            float(summaries[name]["mean_fold_f1"]),
            -int(summaries[name]["false_positives"]),
        ),
    )
    selected_architecture += "_normal_rank"
    report = {
        "comparison_method": (
            "stratified cross-validation within the calibration partition; "
            "thresholds calibrated independently per fold"
        ),
        "folds": folds,
        "threshold_optimization_metric": (
            "maximum recall under the configured strict calibration FPR cap"
        ),
        "model_selection_metric": "mean cross-validated harmonic mean (F1)",
        "maximum_false_positive_rate": args.maximum_false_positive_rate,
        "score_normalization": (
            "empirical within-domain normal-score ranks, with global-reference fallback"
        ),
        "knn_score_normalization": (
            "mean-distance odds adjustment using standardized log-mel features; "
            "optional balanced autoencoder latent block"
        ),
        "autoencoder_feature_stack_candidate": {
            "candidate": "small_knn_autoencoder",
            "feature_blocks": [
                "log-mel spectrogram summary",
                "locally saved autoencoder latent mean and standard deviation",
            ],
            "selection_metric": "mean cross-validated harmonic mean (F1)",
            "secondary_metrics": [
                "standardized partial ROC AUC (max FPR=0.1)",
                "pooled F1",
            ],
        },
        "large_model_f1_wins": model_wins["large"],
        "required_consistent_fold_wins": required_wins,
        "large_model_consistently_better": large_consistent,
        "large_knn_f1_wins": model_wins["large_knn"],
        "large_knn_consistently_better": "large_knn" in consistent_candidates,
        "rbf_svm_f1_wins": model_wins["rbf_svm"],
        "rbf_svm_consistently_better": svm_consistent,
        "rbf_svm_knn_f1_wins": model_wins["rbf_svm_knn"],
        "rbf_svm_knn_consistently_better": (
            "rbf_svm_knn" in consistent_candidates
        ),
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
                "anomaly_score",
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
    use_dual_channel = architecture == "small_dual_channel_normal_rank"
    use_normal_rank = architecture.endswith("_normal_rank")
    architecture_without_rank = (
        architecture.removesuffix("_normal_rank")
        if use_normal_rank
        else architecture
    )
    use_autoencoder_features = architecture_without_rank.endswith("_autoencoder")
    architecture_without_channel_mode = (
        architecture_without_rank.removesuffix("_dual_channel")
        if architecture_without_rank.endswith("_dual_channel")
        else architecture_without_rank
    )
    architecture_without_autoencoder = (
        architecture_without_channel_mode.removesuffix("_autoencoder")
        if architecture_without_channel_mode.endswith("_autoencoder")
        else architecture_without_channel_mode
    )
    model_architecture = architecture_without_autoencoder
    model_architecture = model_architecture.removesuffix("_knn")
    base_architecture = (
        "large"
        if model_architecture in ("large", "large_pruned")
        else "small"
    )
    if model_architecture == "rbf_svm":
        state_dict = checkpoint["state_dict"]
        model = RbfSVMScorer(
            state_dict["support_vectors"].numpy(),
            np.zeros(state_dict["dual_coefficients"].numel(), dtype=np.float32),
            0.0,
            1.0,
        ).to(device)
    else:
        model = FeatureClassifier(
            hidden_dims,
            int(checkpoint.get("input_dim", N_MELS * 10)),
        ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    features = (
        load_dual_channel_features(args.files)
        if use_dual_channel
        else load_features(args.files)[:, None, :, :]
    )
    mean = checkpoint["mean"].numpy()
    std = checkpoint["std"].numpy()
    normalized_features = (
        _normalize_dual_channel(features, mean, std)
        if use_dual_channel
        else _normalize(features, mean, std)
    )
    normalized = (
        _summarize_dual_channel(normalized_features)
        if use_dual_channel
        else _prepare_model_features(normalized_features, base_architecture)
    )
    scores = _probabilities(model, normalized, device)
    knn_features = normalized
    if use_autoencoder_features:
        autoencoder_data = checkpoint.get("autoencoder_feature_extractor")
        if autoencoder_data is None:
            raise ValueError(
                "The checkpoint architecture requires stored autoencoder features."
            )
        autoencoder_extractor = _build_autoencoder_extractor(
            autoencoder_data["state_dict"], device
        )
        autoencoder_features = _autoencoder_features(
            features,
            autoencoder_extractor,
            autoencoder_data["mean"].numpy(),
            autoencoder_data["std"].numpy(),
            device,
        )
        knn_features = _stack_knn_features(normalized, autoencoder_features)
    knn_normalizer = checkpoint.get("knn_normalizer")
    if knn_normalizer is not None:
        stats: dict[str, object] = {
            "mean": knn_normalizer["mean"].to(device),
            "std": knn_normalizer["std"].to(device),
            "reference": knn_normalizer["reference"].to(device),
            "baseline_distance": knn_normalizer["baseline_distance"],
            "neighbors": knn_normalizer["neighbors"],
            "alpha": knn_normalizer["alpha"],
            "block_dims": tuple(knn_normalizer.get("block_dims", ())),
            "block_scales": tuple(knn_normalizer.get("block_scales", ())),
        }
        if not stats["block_dims"]:
            stats["block_dims"] = (knn_features.shape[1],)
            stats["block_scales"] = (1.0,)
        scores = _apply_knn_score_normalization(scores, knn_features, stats, device)
    if use_normal_rank:
        rank_calibrator = checkpoint.get("normal_rank_calibrator")
        if rank_calibrator is None:
            raise ValueError(
                "The checkpoint architecture requires a normal-rank calibrator."
            )
        overall_reference = rank_calibrator["overall_reference"].numpy()
        references_by_domain = {
            domain: reference.numpy()
            for domain, reference in rank_calibrator[
                "references_by_domain"
            ].items()
        }
        scores = _normal_rank_scores(
            scores, args.files, overall_reference, references_by_domain
        )
    threshold = float(checkpoint["threshold"])
    domain_thresholds = checkpoint.get("thresholds_by_domain", {})
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Scoring on {device} ({device_name})")
    for path, score in zip(args.files, scores):
        path_threshold = float(domain_thresholds.get(_domain(path), threshold))
        result: dict[str, str | float] = {
            "file": str(path),
            "threshold": path_threshold,
            "prediction": "anomaly" if score >= path_threshold else "normal",
        }
        if model_architecture == "rbf_svm" or knn_normalizer is not None or use_normal_rank:
            result["anomaly_score"] = float(score)
            result["score_kind"] = (
                "empirical normal-validation rank"
                if use_normal_rank
                else "kNN distance-adjusted anomaly score"
                if knn_normalizer is not None
                else "sigmoid-transformed SVM margin (not calibrated probability)"
            )
        else:
            result["anomaly_probability"] = float(score)
        print(json.dumps(result))


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
    train_parser.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs"
        / "bearing_autoencoder.pt",
        help="saved in-domain autoencoder checkpoint used by the ensemble candidate",
    )
    train_parser.add_argument("--epochs", type=int, default=40)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--early-stopping-patience", type=int, default=7)
    train_parser.add_argument("--evaluation-fraction", type=float, default=0.3)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument(
        "--architecture",
        choices=(
            "small",
            "small_knn",
            "small_calibration_only",
            "small_calibration_only_knn",
            "large",
            "large_knn",
            "large_pruned",
            "large_pruned_knn",
            "rbf_svm",
            "rbf_svm_knn",
            "small_normal_rank",
            "small_knn_normal_rank",
            "small_calibration_only_normal_rank",
            "small_calibration_only_knn_normal_rank",
            "large_normal_rank",
            "large_knn_normal_rank",
            "large_pruned_normal_rank",
            "large_pruned_knn_normal_rank",
            "rbf_svm_normal_rank",
            "rbf_svm_knn_normal_rank",
            "small_knn_autoencoder_normal_rank",
            "small_dual_channel_normal_rank",
        ),
        default="small_normal_rank",
        help="classifier strategy with optional kNN, autoencoder, dual-channel features, and rank scaling",
    )
    train_parser.add_argument("--prune-keep-fraction", type=float, default=0.7)
    train_parser.add_argument(
        "--maximum-false-positive-rate",
        type=float,
        default=0.1,
        help="strict upper bound for calibration normal false-positive rate",
    )
    train_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="compute device (auto uses CUDA when available)",
    )
    train_parser.set_defaults(handler=train)

    compare_parser = commands.add_parser(
        "compare", help="compare model strategies by cross-validated harmonic mean"
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
    compare_parser.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs"
        / "bearing_autoencoder.pt",
        help="saved in-domain autoencoder checkpoint used by the ensemble candidate",
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
        "--maximum-false-positive-rate",
        type=float,
        default=0.1,
        help="strict upper bound for calibration normal false-positive rate",
    )
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
    if args.command in ("train", "compare") and not (
        0 < args.maximum_false_positive_rate < 1
    ):
        raise ValueError("--maximum-false-positive-rate must be between 0 and 1.")
    if args.command in ("train", "compare") and not 0.1 <= args.prune_keep_fraction < 1:
        raise ValueError("--prune-keep-fraction must be in [0.1, 1).")
    args.handler(args)


if __name__ == "__main__":
    main()
