"""Datasets, transforms and DataLoaders built on top of the frozen split.

Also provides the three stress-test manipulations. They run on a PIL image
after Resize + CenterCrop and before ToTensor + Normalize, and are fully
deterministic: the same image always yields the same manipulated result.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Final

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from split_data import PROJECT_ROOT, load_split


IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)

BATCH_SIZE: Final[int] = 64
NUM_WORKERS: Final[int] = 4

# Fixed color-jitter factors — deliberately not random.
JITTER_BRIGHTNESS: Final[float] = 1.3
JITTER_CONTRAST: Final[float] = 1.3
JITTER_SATURATION: Final[float] = 1.4
JITTER_HUE: Final[float] = 0.05

SALT_PEPPER_FRACTION: Final[float] = 0.02

DEBUG_GRID_PATH: Final[Path] = PROJECT_ROOT / "debug_manipulations.png"

Sample = tuple[str, int]
Manipulation = Callable[[Image.Image, str], Image.Image]


# ── dataset ───────────────────────────────────────────────────────────────────

class BirdDataset(Dataset):
    """Reads images listed in the frozen split.

    Args:
        samples: (relative_path, label) tuples from ``load_split()``.
        transform: applied last, turns a PIL image into a normalized tensor.
        pre_transform: optional PIL -> PIL step applied first (the geometric
            part of the val transform, so a manipulation sees the final crop).
        manipulation: optional ``fn(image, relative_path) -> image`` applied
            between ``pre_transform`` and ``transform``.
    """

    def __init__(
        self,
        samples: list[Sample],
        transform=None,
        pre_transform=None,
        manipulation: Manipulation | None = None,
    ):
        self.samples = list(samples)
        self.transform = transform
        self.pre_transform = pre_transform
        self.manipulation = manipulation

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        relative_path, label = self.samples[idx]

        with Image.open(PROJECT_ROOT / relative_path) as image:
            image = image.convert("RGB")

        if self.pre_transform is not None:
            image = self.pre_transform(image)

        if self.manipulation is not None:
            image = self.manipulation(image, relative_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ── transforms ────────────────────────────────────────────────────────────────

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Must match evaluate.py exactly.
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# The same pipeline, split so a manipulation can be inserted in the middle.
val_pre_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

val_post_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ── manipulations ─────────────────────────────────────────────────────────────

def to_black_white(image: Image.Image, path: str = "") -> Image.Image:
    """Grayscale, expanded back to 3 identical RGB channels."""
    return TF.to_grayscale(image, num_output_channels=3)


def to_color_jitter(image: Image.Image, path: str = "") -> Image.Image:
    """Fixed brightness / contrast / saturation / hue shift (never random)."""
    image = TF.adjust_brightness(image, JITTER_BRIGHTNESS)
    image = TF.adjust_contrast(image, JITTER_CONTRAST)
    image = TF.adjust_saturation(image, JITTER_SATURATION)
    image = TF.adjust_hue(image, JITTER_HUE)
    return image


def _seed_from_path(path: str) -> int:
    """Stable 64-bit seed for a file path, identical across runs and machines."""
    digest = hashlib.sha256(path.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def to_salt_pepper(
    image: Image.Image,
    path: str = "",
    fraction: float = SALT_PEPPER_FRACTION,
) -> Image.Image:
    """Set ``fraction``/2 of pixels to pure black and as many to pure white.

    The noise pattern is derived from a hash of the file path, so each image
    keeps its own fixed pattern for good.
    """
    array = np.array(image.convert("RGB"))
    height, width = array.shape[:2]

    n_pixels = height * width
    n_each = int(round(n_pixels * fraction / 2))

    if n_each > 0:
        rng = np.random.default_rng(_seed_from_path(path))
        chosen = rng.choice(n_pixels, size=2 * n_each, replace=False)

        pepper = np.unravel_index(chosen[:n_each], (height, width))
        salt = np.unravel_index(chosen[n_each:], (height, width))

        array[pepper] = 0
        array[salt] = 255

    return Image.fromarray(array)


MANIPULATIONS: Final[dict[str, Manipulation]] = {
    "black_white": to_black_white,
    "color_jitter": to_color_jitter,
    "salt_pepper": to_salt_pepper,
}


# ── loaders ───────────────────────────────────────────────────────────────────

def get_train_loader(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> DataLoader:
    """Shuffled loader over the 16000 training images."""
    train_samples, _ = load_split()

    dataset = BirdDataset(train_samples, transform=train_transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )


def get_val_loader(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> DataLoader:
    """Unshuffled loader over the 4000 clean validation images."""
    _, val_samples = load_split()

    dataset = BirdDataset(val_samples, transform=val_transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def get_stress_loader(
    name: str,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> DataLoader:
    """Unshuffled validation loader with one manipulation applied."""
    if name not in MANIPULATIONS:
        available = ", ".join(sorted(MANIPULATIONS))
        raise KeyError(f"Unknown manipulation {name!r}. Available: {available}")

    _, val_samples = load_split()

    dataset = BirdDataset(
        val_samples,
        transform=val_post_transform,
        pre_transform=val_pre_transform,
        manipulation=MANIPULATIONS[name],
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


# ── visual check ──────────────────────────────────────────────────────────────

def _save_manipulation_grid(
    val_samples: list[Sample],
    n_images: int = 3,
    output_path: Path = DEBUG_GRID_PATH,
) -> Path:
    """Save a grid: one row per image, columns = original + each manipulation."""
    from PIL import ImageDraw

    names = ["original", *MANIPULATIONS]

    cell = 224
    pad = 8
    header = 20

    # Spread the picks across classes instead of taking three of the same bird.
    step = max(1, len(val_samples) // n_images)
    picks = [val_samples[i * step] for i in range(n_images)]

    width = len(names) * cell + (len(names) + 1) * pad
    height = header + n_images * cell + (n_images + 1) * pad

    grid = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(grid)

    for col, name in enumerate(names):
        x = pad + col * (cell + pad)
        draw.text((x, 4), name, fill="black")

    for row, (relative_path, _) in enumerate(picks):
        with Image.open(PROJECT_ROOT / relative_path) as image:
            image = image.convert("RGB")

        base = val_pre_transform(image)

        for col, name in enumerate(names):
            if name == "original":
                view = base
            else:
                view = MANIPULATIONS[name](base, relative_path)

            x = pad + col * (cell + pad)
            y = header + pad + row * (cell + pad)
            grid.paste(view, (x, y))

    grid.save(output_path)
    return output_path


if __name__ == "__main__":
    train_samples, val_samples = load_split()

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples:   {len(val_samples)}")

    train_loader = get_train_loader(batch_size=8, num_workers=0)
    images, labels = next(iter(train_loader))

    print(
        f"\nTrain batch: images {tuple(images.shape)} {images.dtype}, "
        f"labels {tuple(labels.shape)} {labels.dtype}"
    )

    for name in MANIPULATIONS:
        stress_loader = get_stress_loader(name, batch_size=8, num_workers=0)
        stress_images, _ = next(iter(stress_loader))
        print(f"Stress batch [{name}]: {tuple(stress_images.shape)} {stress_images.dtype}")

    path = _save_manipulation_grid(val_samples)
    print(f"\nWrote {path}")
