"""Benchmark PyTorch and Keras model checkpoints and publish machine-readable telemetry."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

# Suppress verbose TensorFlow startup logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RESULTS_PATH = ROOT / "data" / "results.json"
WARMUP_PASSES = 10
TIMED_PASSES = 50
THRESHOLDS = {
    "max_latency_ms": 5.0,
    "max_p95_latency_ms": 500.0,
    "max_model_size_mb": 100.0,
    "max_memory_rss_mb": 512.0,
}


class DummyCNN(nn.Module):
    """Architecture used by the portable baseline checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(8 * 4 * 4, 2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _benchmark_pytorch(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
        architecture = model.__class__.__name__.lower()
        input_shape = [1, 1, 28, 28] if "conv" in architecture or "cnn" in architecture else [1, 32]
        modality = "image" if len(input_shape) == 4 else "tabular"
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), nn.Module):
        model = checkpoint["model"]
        architecture = str(checkpoint.get("architecture", model.__class__.__name__)).lower()
        input_shape = list(checkpoint.get("input_shape", [1, 1, 28, 28]))
        modality = str(checkpoint.get("modality", "image" if "cnn" in architecture else "sequence"))
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        architecture = str(checkpoint.get("architecture", "")).lower()
        if architecture in {"dummy_cnn", "cnn"}:
            model = DummyCNN()
        else:
            raise ValueError(f"Unsupported state-dict architecture: {architecture or 'unspecified'}")
        model.load_state_dict(checkpoint["state_dict"])
        input_shape = list(checkpoint.get("input_shape", [1, 1, 28, 28]))
        modality = str(checkpoint.get("modality", "image" if len(input_shape) == 4 else "sequence"))
    else:
        raise ValueError(f"{path.name} must contain a serialized torch.nn.Module")

    model.eval()
    inputs = torch.randn(*input_shape)
    process = psutil.Process(os.getpid())
    before_rss = process.memory_info().rss

    with torch.inference_mode():
        for _ in range(WARMUP_PASSES):
            model(inputs)
        latencies: list[float] = []
        for _ in range(TIMED_PASSES):
            started = time.perf_counter()
            model(inputs)
            latencies.append((time.perf_counter() - started) * 1000)

    after_rss = process.memory_info().rss
    file_size_mb = path.stat().st_size / (1024 * 1024)
    rss_mb = max(after_rss - before_rss, 0) / (1024 * 1024)
    mean_latency = sum(latencies) / len(latencies)
    p95_latency = _percentile(latencies, 95)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    return {
        "model": path.name,
        "path": str(path.relative_to(ROOT)),
        "modality": modality,
        "architecture": model.__class__.__name__,
        "input_shape": input_shape,
        "file_size_mb": round(file_size_mb, 4),
        "memory_rss_mb": round(rss_mb, 4),
        "mean_latency_ms": round(mean_latency, 4),
        "p95_latency_ms": round(p95_latency, 4),
        "throughput_fps": round(1000 / mean_latency, 4) if mean_latency else 0.0,
        "parameter_count": parameter_count,
        "thresholds": THRESHOLDS,
        "validation": {
            "latency": mean_latency < THRESHOLDS["max_latency_ms"],
            "p95_latency": p95_latency < THRESHOLDS["max_p95_latency_ms"],
            "file_size": file_size_mb < THRESHOLDS["max_model_size_mb"],
            "memory": rss_mb < THRESHOLDS["max_memory_rss_mb"],
        },
    }


def _benchmark_keras(path: Path) -> dict[str, Any]:
    import tensorflow as tf

    model = tf.keras.models.load_model(str(path), compile=False)
    
    # Exact shape matching your Gradio app.py preprocessing: 150x150x3
    input_shape = [1, 150, 150, 3]
    inputs = tf.random.normal(input_shape)

    process = psutil.Process(os.getpid())
    before_rss = process.memory_info().rss

    # Warmup passes
    for _ in range(WARMUP_PASSES):
        _ = model(inputs, training=False)

    latencies: list[float] = []
    for _ in range(TIMED_PASSES):
        started = time.perf_counter()
        _ = model(inputs, training=False)
        latencies.append((time.perf_counter() - started) * 1000)

    after_rss = process.memory_info().rss
    file_size_mb = path.stat().st_size / (1024 * 1024)
    rss_mb = max(after_rss - before_rss, 0) / (1024 * 1024)
    mean_latency = sum(latencies) / len(latencies)
    p95_latency = _percentile(latencies, 95)
    parameter_count = model.count_params()

    return {
        "model": path.name,
        "path": str(path.relative_to(ROOT)),
        "modality": "image (keras cnn)",
        "architecture": model.__class__.__name__,
        "input_shape": input_shape,
        "file_size_mb": round(file_size_mb, 4),
        "memory_rss_mb": round(rss_mb, 4),
        "mean_latency_ms": round(mean_latency, 4),
        "p95_latency_ms": round(p95_latency, 4),
        "throughput_fps": round(1000 / mean_latency, 4) if mean_latency else 0.0,
        "parameter_count": parameter_count,
        "thresholds": THRESHOLDS,
        "validation": {
            "latency": mean_latency < THRESHOLDS["max_latency_ms"],
            "p95_latency": p95_latency < THRESHOLDS["max_p95_latency_ms"],
            "file_size": file_size_mb < THRESHOLDS["max_model_size_mb"],
            "memory": rss_mb < THRESHOLDS["max_memory_rss_mb"],
        },
    }


def _benchmark(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth", ".bin"}:
        return _benchmark_pytorch(path)
    elif suffix == ".keras":
        return _benchmark_keras(path)
    else:
        raise ValueError(f"Unsupported model extension: {suffix}")


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_paths = sorted(
        path for path in MODELS_DIR.iterdir() if path.suffix.lower() in {".pt", ".pth", ".bin", ".keras"}
    )
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in model_paths:
        try:
            results.append(_benchmark(path))
        except Exception as error:
            errors.append({"model": path.name, "error": str(error)})

    passed = bool(results) and not errors and all(
        all(model["validation"].values()) for model in results
    )
    payload = {
        "status": "PASSED" if passed else "FAILED",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "warmup_passes": WARMUP_PASSES,
        "timed_passes": TIMED_PASSES,
        "thresholds": THRESHOLDS,
        "models": results,
        "errors": errors,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())