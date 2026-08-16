"""Model architectures for Cats vs Dogs binary classification.

Two baselines are available, selected via the `arch` key in params.yaml:
  - "simple_cnn": a small CNN trained from scratch
  - "resnet18":   ImageNet-pretrained ResNet18 with a frozen backbone and a
                  retrained classifier head (transfer learning)
"""

import torch
from torch import nn
from torchvision import models


class SimpleCNN(nn.Module):
    """Small 4-block CNN for 224x224 RGB input, 2 output classes."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 128),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ResNet18Transfer(nn.Module):
    """ImageNet-pretrained ResNet18 with a frozen backbone and a new head.

    Only the final fully-connected layer is trained, which converges in a few
    epochs on CPU and far outperforms the from-scratch CNN on this dataset.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def train(self, mode: bool = True):
        """Keep the frozen backbone in eval mode so its BatchNorm stats stay fixed."""
        super().train(mode)
        if mode:
            self.backbone.eval()
            self.backbone.fc.train()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(arch: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Instantiate an architecture by name."""
    if arch == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    if arch == "resnet18":
        return ResNet18Transfer(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown arch {arch!r}; expected one of ['resnet18', 'simple_cnn']")


def save_model(
    model: nn.Module, path: str, class_names: list[str], img_size: int, arch: str
) -> None:
    """Serialize model weights plus the metadata inference needs."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "img_size": img_size,
            "arch": arch,
        },
        path,
    )


def load_model(path: str, device: str = "cpu") -> tuple[nn.Module, list[str], int]:
    """Load a checkpoint saved by save_model. Returns (model, class_names, img_size)."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    # Weights come from the checkpoint, so skip downloading pretrained ones.
    model = build_model(
        checkpoint["arch"],
        num_classes=len(checkpoint["class_names"]),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint["class_names"], checkpoint["img_size"]
