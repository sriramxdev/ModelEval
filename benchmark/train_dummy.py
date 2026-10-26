"""Create a small, deterministic CNN checkpoint for local and CI evaluation."""

from pathlib import Path

import torch
from torch import nn


class DummyCNN(nn.Module):
    """A compact image classifier used as a reproducible benchmark fixture."""

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


def main() -> None:
    torch.manual_seed(42)
    model = DummyCNN().eval()
    output_path = Path(__file__).resolve().parents[1] / "models" / "baseline_model.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": "dummy_cnn",
            "input_shape": [1, 1, 28, 28],
            "modality": "image",
        },
        output_path,
    )
    print(f"Saved baseline model to {output_path}")


if __name__ == "__main__":
    main()
