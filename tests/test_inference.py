"""Unit tests for model utilities and the inference API (M3 requirement)."""

import io

import pytest
import torch
from PIL import Image

from src.models.model import ResNet18Transfer, SimpleCNN, build_model, load_model, save_model
from src.utils import CLASS_NAMES


def test_forward_pass_shape_and_valid_probabilities():
    model = SimpleCNN(num_classes=2)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(4, 3, 224, 224))
        probs = torch.softmax(logits, dim=1)

    assert logits.shape == (4, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_save_and_load_model_roundtrip(tmp_path):
    model = SimpleCNN(num_classes=2)
    path = tmp_path / "model.pt"
    save_model(model, str(path), CLASS_NAMES, img_size=224, arch="simple_cnn")

    loaded, class_names, img_size = load_model(str(path))
    assert class_names == CLASS_NAMES
    assert img_size == 224

    x = torch.randn(1, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-6)


def test_resnet_backbone_is_frozen_except_head():
    model = ResNet18Transfer(num_classes=2, pretrained=False)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]

    assert trainable, "classifier head must remain trainable"
    assert all(name.startswith("backbone.fc") for name in trainable)
    # Frozen BatchNorm: backbone must stay in eval mode even after .train().
    model.train()
    assert not model.backbone.layer1.training
    assert model.backbone.fc.training


def test_save_and_load_preserves_architecture(tmp_path):
    path = tmp_path / "resnet.pt"
    save_model(
        ResNet18Transfer(num_classes=2, pretrained=False),
        str(path),
        CLASS_NAMES,
        img_size=224,
        arch="resnet18",
    )

    loaded, _, _ = load_model(str(path))
    assert isinstance(loaded, ResNet18Transfer)


def test_build_model_rejects_unknown_arch():
    with pytest.raises(ValueError, match="Unknown arch"):
        build_model("not_a_real_arch")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient with a freshly saved (untrained) model on disk."""
    from fastapi.testclient import TestClient

    import app.main as main

    model_path = tmp_path / "model.pt"
    save_model(
        SimpleCNN(num_classes=2), str(model_path), CLASS_NAMES, img_size=224, arch="simple_cnn"
    )
    monkeypatch.setattr(main, "MODEL_PATH", model_path)

    with TestClient(main.app) as test_client:
        yield test_client


def jpeg_bytes(color=(200, 150, 100)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color).save(buf, "JPEG")
    return buf.getvalue()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_endpoint_returns_label_and_probabilities(client):
    response = client.post(
        "/predict", files={"file": ("cat.jpg", jpeg_bytes(), "image/jpeg")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in CLASS_NAMES
    assert set(body["probabilities"]) == set(CLASS_NAMES)
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01


def test_predict_rejects_non_image(client):
    response = client.post(
        "/predict", files={"file": ("bad.jpg", b"not an image", "image/jpeg")}
    )
    assert response.status_code == 400
