from __future__ import annotations

import argparse
import copy
import csv
import json
import random
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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


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
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(N_MELS * 10, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(values)).squeeze(1)


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


def _recall_at_minimum_precision_threshold(
    labels: np.ndarray, scores: np.ndarray, minimum_precision: float
) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        raise ValueError("Could not calibrate a threshold from the labeled clips.")
    feasible = np.flatnonzero(precision[:-1] >= minimum_precision)
    if feasible.size == 0:
        raise ValueError(
            f"No calibration threshold achieved the requested precision "
            f"of {minimum_precision:.3f}."
        )
    best_recall = recall[feasible].max()
    best_candidates = feasible[recall[feasible] == best_recall]
    best_index = best_candidates[np.argmax(precision[best_candidates])]
    return float(thresholds[best_index])


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
    return {
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "confusion_matrix_normal_anomaly": matrix.tolist(),
        "samples": int(len(labels)),
        "anomalies": int(labels.sum()),
    }


def train(args: argparse.Namespace) -> None:
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
    train_features = _summarize_spectrograms(_normalize(train_features, mean, std))
    test_features = _summarize_spectrograms(_normalize(test_features, mean, std))

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
    fit_features = test_features[classifier_train_indices]
    fit_labels = test_labels[classifier_train_indices]
    fit_indices, validation_indices = train_test_split(
        np.arange(len(fit_labels)),
        test_size=0.15,
        random_state=args.seed,
        stratify=fit_labels,
    )
    train_tensor = torch.from_numpy(fit_features[fit_indices])
    train_labels_tensor = torch.from_numpy(fit_labels[fit_indices]).float()
    validation_tensor = torch.from_numpy(fit_features[validation_indices])
    validation_labels = fit_labels[validation_indices]
    loader = DataLoader(
        TensorDataset(train_tensor, train_labels_tensor),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    device = _get_device(args.device)
    model = FeatureClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    positive_count = int(fit_labels[fit_indices].sum())
    negative_count = len(fit_indices) - positive_count
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [negative_count / max(positive_count, 1)], device=device
        )
    )
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    patience = max(5, args.early_stopping_patience)

    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(
        f"Training deep classifier on {device} ({device_name}) using {len(train_paths)} normal "
        f"clips for normalization and {len(classifier_train_indices)} labeled "
        "calibration clips..."
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch, batch_labels in loader:
            batch = batch.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch), batch_labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_tensor.to(device))
            validation_loss = float(
                loss_function(
                    validation_logits,
                    torch.from_numpy(validation_labels.astype(np.float32)).to(device),
                ).item()
            )
        print(f"Epoch {epoch:02d}/{args.epochs} - val_loss={validation_loss:.6f}")
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    threshold_scores = _probabilities(
        model, test_features[threshold_indices], device
    )
    threshold = _recall_at_minimum_precision_threshold(
        test_labels[threshold_indices], threshold_scores, args.minimum_precision
    )
    domain_thresholds: dict[str, float] = {}
    for domain in ("source", "target"):
        mask = np.array(
            [_domain(test_paths[index]) == domain for index in threshold_indices]
        )
        if mask.any() and np.unique(test_labels[threshold_indices][mask]).size == 2:
            domain_thresholds[domain] = _recall_at_minimum_precision_threshold(
                test_labels[threshold_indices][mask],
                threshold_scores[mask],
                args.minimum_precision,
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
    threshold_validation_thresholds = np.array(
        [
            domain_thresholds.get(_domain(test_paths[index]), threshold)
            for index in threshold_indices
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
        "threshold": threshold,
        "thresholds_by_domain": domain_thresholds,
        "threshold_optimization_metric": (
            f"maximum recall with precision >= {args.minimum_precision:.3f}"
        ),
        "minimum_precision": args.minimum_precision,
        "threshold_calibrated_on": (
            "labeled threshold-validation subset of the calibration split"
        ),
        "evaluation": _metrics(evaluation_labels, evaluation_predictions),
        "threshold_validation": _metrics(
            test_labels[threshold_indices], threshold_predictions
        ),
        "evaluation_meets_minimum_precision": bool(
            precision_score(
                evaluation_labels, evaluation_predictions, zero_division=0
            )
            >= args.minimum_precision
        ),
        "threshold_validation_meets_minimum_precision": bool(
            precision_score(
                test_labels[threshold_indices],
                threshold_predictions,
                zero_division=0,
            )
            >= args.minimum_precision
        ),
        "normalization_normal_clips": len(train_paths),
        "labeled_calibration_training_clips": len(classifier_train_indices),
        "test_clips": len(test_paths),
        "evaluation_fraction": args.evaluation_fraction,
        "seed": args.seed,
        "device": str(device),
    }

    checkpoint_path = output_dir / "bearing_classifier.pt"
    torch.save(
        {
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
    print(
        f"Held-out precision target met: "
        f"{report['evaluation_meets_minimum_precision']}"
    )
    print(
        f"Calibration recall at precision >= {args.minimum_precision:.2f}: "
        f"{report['threshold_validation']['recall']:.3f}"
    )
    print(f"Threshold: {threshold:.8f}")
    print(f"Saved model: {checkpoint_path}")
    print(f"Saved metrics: {metrics_path}")


def predict(args: argparse.Namespace) -> None:
    device = _get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = FeatureClassifier().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    features = load_features(args.files)[:, None, :, :]
    mean = checkpoint["mean"].numpy()
    std = checkpoint["std"].numpy()
    normalized = _summarize_spectrograms(_normalize(features, mean, std))
    scores = _probabilities(model, normalized, device)
    threshold = float(checkpoint["threshold"])
    domain_thresholds = checkpoint["thresholds_by_domain"]
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
        "--minimum-precision",
        type=float,
        default=0.7,
        help="minimum precision required when choosing the recall-maximizing threshold",
    )
    train_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="compute device (auto uses CUDA when available)",
    )
    train_parser.set_defaults(handler=train)

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
    if args.command == "train" and not 0.1 <= args.evaluation_fraction <= 0.8:
        raise ValueError("--evaluation-fraction must be between 0.1 and 0.8.")
    if args.command == "train" and not 0 < args.minimum_precision <= 1:
        raise ValueError("--minimum-precision must be greater than 0 and at most 1.")
    args.handler(args)


if __name__ == "__main__":
    main()
