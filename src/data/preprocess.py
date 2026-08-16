"""Preprocess the raw Cats vs Dogs dataset.

Reads class folders from data/raw (any folder whose name contains "cat" or
"dog", case-insensitively, at any depth), filters out corrupt/unreadable
images, resizes everything to 224x224 RGB JPEGs, and writes a stratified
80/10/10 train/val/test split to data/processed/{train,val,test}/{cat,dog}.

Usage:
    python -m src.data.preprocess [--raw-dir data/raw] [--out-dir data/processed]
"""

import argparse
import random
from pathlib import Path

import yaml
from PIL import Image

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_valid_image(path: Path) -> bool:
    """Return True if the file is a readable, decodable image."""
    try:
        with Image.open(path) as img:
            img.verify()
        # verify() leaves the file unusable; reopen to force a full decode.
        with Image.open(path) as img:
            img.convert("RGB")
        return True
    except Exception:
        return False


def find_class_images(raw_dir: Path) -> dict[str, list[Path]]:
    """Map class name -> image paths by scanning folder names for cat/dog."""
    images: dict[str, list[Path]] = {"cat": [], "dog": []}
    for path in raw_dir.rglob("*"):
        if not (path.is_file() and path.suffix.lower() in IMG_EXTENSIONS):
            continue
        folder = path.parent.name.lower()
        if "cat" in folder:
            images["cat"].append(path)
        elif "dog" in folder:
            images["dog"].append(path)
    return images


def split_indices(n: int, train_ratio: float, val_ratio: float, seed: int):
    """Return shuffled index lists (train, val, test) covering range(n)."""
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (
        indices[:n_train],
        indices[n_train : n_train + n_val],
        indices[n_train + n_val :],
    )


def process_and_save(src: Path, dest: Path, img_size: int) -> bool:
    """Resize one image to img_size RGB and save as JPEG. False on failure."""
    try:
        with Image.open(src) as img:
            img.convert("RGB").resize((img_size, img_size)).save(dest, "JPEG", quality=90)
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    with open(args.params) as f:
        cfg = yaml.safe_load(f)["preprocess"]

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    class_images = find_class_images(raw_dir)
    if not any(class_images.values()):
        raise SystemExit(f"No cat/dog images found under {raw_dir}")

    max_per_class = cfg.get("max_per_class")
    total_skipped = 0
    for cls, paths in class_images.items():
        if max_per_class:
            # Deterministic subsample before validation to keep runtime bounded.
            paths = sorted(paths)
            random.Random(cfg["seed"]).shuffle(paths)
            paths = paths[: max_per_class + 200]  # headroom for corrupt files
        valid = [p for p in paths if is_valid_image(p)]
        if max_per_class:
            valid = valid[:max_per_class]
        total_skipped += len(paths) - len(valid)
        print(f"{cls}: {len(valid)} valid images ({len(paths) - len(valid)} corrupt skipped)")

        train_idx, val_idx, test_idx = split_indices(
            len(valid), cfg["train_ratio"], cfg["val_ratio"], cfg["seed"]
        )
        for split, idx_list in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            split_dir = out_dir / split / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            for i in idx_list:
                dest = split_dir / f"{cls}_{i:05d}.jpg"
                if not process_and_save(valid[i], dest, cfg["img_size"]):
                    total_skipped += 1
            print(f"  {split}: {len(idx_list)} images")

    print(f"Done. Output: {out_dir} (total skipped: {total_skipped})")


if __name__ == "__main__":
    main()
