"""Unit tests for data pre-processing functions (M3 requirement)."""

from pathlib import Path

from PIL import Image

from src.data.preprocess import is_valid_image, process_and_save, split_indices
from src.utils import get_eval_transforms, get_train_transforms


def make_image(path: Path, size=(320, 200), color=(120, 80, 40)) -> Path:
    Image.new("RGB", size, color).save(path, "JPEG")
    return path


def test_is_valid_image_accepts_good_and_rejects_corrupt(tmp_path):
    good = make_image(tmp_path / "good.jpg")
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"this is not an image")

    assert is_valid_image(good) is True
    assert is_valid_image(corrupt) is False


def test_process_and_save_resizes_to_target_rgb(tmp_path):
    src = make_image(tmp_path / "src.jpg", size=(500, 300))
    dest = tmp_path / "out.jpg"

    assert process_and_save(src, dest, img_size=224) is True
    with Image.open(dest) as img:
        assert img.size == (224, 224)
        assert img.mode == "RGB"


def test_split_indices_ratios_and_coverage():
    train, val, test = split_indices(100, train_ratio=0.8, val_ratio=0.1, seed=42)

    assert len(train) == 80 and len(val) == 10 and len(test) == 10
    # Every index appears exactly once across the three splits (no leakage).
    assert sorted(train + val + test) == list(range(100))


def test_split_indices_deterministic_for_same_seed():
    assert split_indices(50, 0.8, 0.1, seed=7) == split_indices(50, 0.8, 0.1, seed=7)
    assert split_indices(50, 0.8, 0.1, seed=7) != split_indices(50, 0.8, 0.1, seed=8)


def test_transforms_produce_normalized_tensor_of_expected_shape(tmp_path):
    img = Image.new("RGB", (300, 400), (100, 150, 200))

    for transform in (get_train_transforms(224), get_eval_transforms(224)):
        tensor = transform(img)
        assert tuple(tensor.shape) == (3, 224, 224)
        assert tensor.dtype.is_floating_point
