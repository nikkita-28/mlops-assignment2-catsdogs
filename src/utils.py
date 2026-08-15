"""Shared constants and transforms for training and inference."""

from torchvision import transforms

CLASS_NAMES = ["cat", "dog"]
IMG_SIZE = 224

# ImageNet statistics — standard normalization for 224x224 RGB CNN inputs.
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Augmented pipeline used only on the training split."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )


def get_eval_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Deterministic pipeline for validation, test, and inference."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )
