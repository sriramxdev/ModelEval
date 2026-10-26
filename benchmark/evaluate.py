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
CONFIG_PATH = ROOT / "configs" / "eval_config.json"
RESULTS_PATH = ROOT / "data" / "results.json"
WARMUP_PASSES = 10
TIMED_PASSES = 50

# Fallback global thresholds if config doesn't specify them
DEFAULT_THRESHOLDS = {
    "max_latency_ms": 5.0,
    "max_p95_latency_ms": 500.0,
    "max_model_size_mb": 100.0,
    "max_memory_rss_mb": 512.0,
}

# Load evaluation config registry safely
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r") as f:
        EVAL_CONFIG = json.load(f)
else:
    EVAL_CONFIG = {}


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


def _get_spec_for_model(filename: str) -> dict[str, Any]:
    """Matches filename substrings to eval_config.json specifications."""
    filename_lower = filename.lower()
    for key, spec in EVAL_CONFIG.items():
        if spec.get("match_pattern", "") in filename_lower:
            return spec
    # Fallback default spec if no pattern matches
    return {
        "modality": "image",
        "input_shape": [1, 1, 28, 28],
        "thresholds": DEFAULT_THRESHOLDS
    }


def _benchmark_pytorch(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    spec = _get_spec_for_model(path.name)
    input_shape = spec["input_shape"]
    modality = spec["modality"]
    thresholds = spec.get("thresholds", DEFAULT_THRESHOLDS)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), nn.Module):
        model = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        architecture = str(checkpoint.get("architecture", "")).lower()
        if architecture in {"dummy_cnn", "cnn"}:
            model = DummyCNN()
        else:
            model = DummyCNN() # Fallback
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        raise ValueError(f"{path.name} must contain a serialized torch.nn.Module")

    model.eval()
    
    if spec.get("dtype") == "long":
        inputs = torch.randint(0, 1000, input_shape, dtype=torch.long)
    else:
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
        "thresholds": thresholds,
        "validation": {
            "latency": mean_latency < thresholds["max_latency_ms"],
            "p95_latency": p95_latency < thresholds["max_p95_latency_ms"],
            "file_size": file_size_mb < thresholds["max_model_size_mb"],
            "memory": rss_mb < thresholds["max_memory_rss_mb"],
        },
    }


def _benchmark_keras(path: Path) -> dict[str, Any]:
    import tensorflow as tf

    model = tf.keras.models.load_model(str(path), compile=False)
    
    spec = _get_spec_for_model(path.name)
    input_shape = spec["input_shape"]
    thresholds = spec.get("thresholds", DEFAULT_THRESHOLDS)
    inputs = tf.random.normal(input_shape)

    process = psutil.Process(os.getpid())
    before_rss = process.memory_info().rss

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
        "modality": spec["modality"],
        "architecture": model.__class__.__name__,
        "input_shape": input_shape,
        "file_size_mb": round(file_size_mb, 4),
        "memory_rss_mb": round(rss_mb, 4),
        "mean_latency_ms": round(mean_latency, 4),
        "p95_latency_ms": round(p95_latency, 4),
        "throughput_fps": round(1000 / mean_latency, 4) if mean_latency else 0.0,
        "parameter_count": parameter_count,
        "thresholds": thresholds,
        "validation": {
            "latency": mean_latency < thresholds["max_latency_ms"],
            "p95_latency": p95_latency < thresholds["max_p95_latency_ms"],
            "file_size": file_size_mb < thresholds["max_model_size_mb"],
            "memory": rss_mb < thresholds["max_memory_rss_mb"],
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
        "thresholds": DEFAULT_THRESHOLDS,
        "models": results,
        "errors": errors,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())