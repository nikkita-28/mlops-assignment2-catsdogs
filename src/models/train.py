"""Train the classifier with MLflow experiment tracking.

The architecture is chosen by the `train.arch` key in params.yaml — either the
from-scratch "simple_cnn" baseline or "resnet18" transfer learning.

Logs params, per-epoch train/val loss and accuracy, final test metrics,
a confusion matrix, loss curves, and the serialized model artifact.

Usage:
    python -m src.models.train [--data-dir data/processed] [--params params.yaml]
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import torch
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from src.models.model import build_model, save_model
from src.utils import CLASS_NAMES, get_eval_transforms, get_train_transforms

# MLflow 3.x refuses the default './mlruns' file store, so run metadata is kept
# in SQLite. Artifacts (plots, checkpoints) still land under mlruns/.
TRACKING_URI = "sqlite:///mlflow.db"


def make_loader(data_dir: Path, split: str, transform, batch_size: int, workers: int, shuffle: bool):
    dataset = datasets.ImageFolder(data_dir / split, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    """One pass over loader. Trains if optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, preds, targets = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds.extend(logits.argmax(1).cpu().tolist())
            targets.extend(y.cpu().tolist())
    return total_loss / len(loader.dataset), accuracy_score(targets, preds), preds, targets


def log_plots(history: dict, test_preds, test_targets, artifact_dir: Path) -> None:
    """Save loss-curve and confusion-matrix figures and log them to MLflow."""
    artifact_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set(title="Loss", xlabel="epoch", ylabel="loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    axes[1].legend()
    curves_path = artifact_dir / "loss_curves.png"
    fig.tight_layout()
    fig.savefig(curves_path)
    plt.close(fig)

    cm = confusion_matrix(test_targets, test_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Test confusion matrix")
    cm_path = artifact_dir / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(cm_path)
    plt.close(fig)

    mlflow.log_artifact(str(curves_path))
    mlflow.log_artifact(str(cm_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--model-out", default="models/model.pt")
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)
    cfg = params["train"]
    img_size = params["preprocess"]["img_size"]

    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_dir)

    train_loader = make_loader(
        data_dir, "train", get_train_transforms(img_size), cfg["batch_size"], cfg["num_workers"], True
    )
    val_loader = make_loader(
        data_dir, "val", get_eval_transforms(img_size), cfg["batch_size"], cfg["num_workers"], False
    )
    test_loader = make_loader(
        data_dir, "test", get_eval_transforms(img_size), cfg["batch_size"], cfg["num_workers"], False
    )

    arch = cfg["arch"]
    model = build_model(arch, num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    if not os.environ.get("MLFLOW_TRACKING_URI"):  # an explicit override wins
        mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(cfg["experiment_name"])
    with mlflow.start_run():
        mlflow.log_params(
            {
                **{k: v for k, v in cfg.items() if k != "experiment_name"},
                "img_size": img_size,
                "model": arch,
                "trainable_params": sum(p.numel() for p in trainable),
                "total_params": sum(p.numel() for p in model.parameters()),
                "device": device,
                "train_size": len(train_loader.dataset),
                "val_size": len(val_loader.dataset),
                "test_size": len(test_loader.dataset),
            }
        )

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        for epoch in range(1, cfg["epochs"] + 1):
            train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion, device=device)
            epoch_metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            for key, value in epoch_metrics.items():
                history[key].append(value)
            mlflow.log_metrics(epoch_metrics, step=epoch)
            print(
                f"epoch {epoch}/{cfg['epochs']} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f}"
            )

        test_loss, test_acc, test_preds, test_targets = run_epoch(
            model, test_loader, criterion, device=device
        )
        test_f1 = f1_score(test_targets, test_preds)
        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc, "test_f1": test_f1})
        print(f"test | loss {test_loss:.4f} acc {test_acc:.4f} f1 {test_f1:.4f}")

        log_plots(history, test_preds, test_targets, Path("reports"))

        model_out = Path(args.model_out)
        model_out.parent.mkdir(parents=True, exist_ok=True)
        save_model(model.cpu(), str(model_out), CLASS_NAMES, img_size, arch=arch)
        mlflow.log_artifact(str(model_out))
        print(f"Model saved to {model_out}")


if __name__ == "__main__":
    main()
