"""Training script that produced weights.joblib.

Self-contained: the frozen split, the datasets, the transforms and the
stress-test manipulations are all defined here, so this file alone documents
how the submitted weights were produced. (During development the same code
lived in split_data.py and data.py alongside this script.)

The label mapping is imported from labels.py in the project root, the same
starter file predict.py reaches through base_model.py.

Data layout expected under BIRD_DATA_ROOT (default: ./dataset):
    <root>/train/<class_name>/*.jpg      or  <root>/train/<idx>_<class_name>/*.jpg

Run:
  python train.py
  python train.py --epochs 1                                  # sanity run
  python train.py --robust                                    # experiment 1
  python train.py --robust --out weights_robust.joblib

Experiment log (30 epochs each, accuracy % on our held-out validation split):

    set            baseline   exp 1   exp 2   exp 3
    clean             79.97   80.03   79.97   78.40
    black_white       34.60   69.17   68.93   72.75
    color_jitter      76.15   79.10   78.83   77.82
    salt_pepper       43.43   32.50   23.20   13.33
    stress mean       51.39   60.26   56.98   54.63

  baseline  train_transform, no manipulation at all
  exp 1     train_transform_robust as defined below
  exp 2     exp 1 + RandomGaussianNoise (kept below, unused) — reverted
  exp 3     exp 1 with grayscale p=0.5, jitter p=0.7 and wider factors — reverted

Salt and pepper is never applied to training data. It is held out as an honest
probe for a manipulation the model has never seen.
"""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Callable, Final

import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from labels import (
    HF_INDEX_TO_NAME,
    HF_INDEX_TO_IDX,
    TARGET_HF_INDICES,
)

from model import ModelArchitecture


# ── config ────────────────────────────────────────────────────────────────────

EPOCHS = 30
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

VAL_FRACTION = 0.2
SEED = 42

# Relative by design — no absolute paths anywhere in the submission. Point
# BIRD_DATA_ROOT at the image folder if it lives outside the project.
DATA_ROOT = Path(os.environ.get("BIRD_DATA_ROOT", "dataset"))
SPLIT_PATH = Path("split.json")

OUTPUT_NAME = "weights.joblib"
OUTPUT_DIR = Path(__file__).resolve().parent

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_PATTERNS: Final[tuple[str, ...]] = ("*.jpg", "*.jpeg", "*.JPEG", "*.png")

SALT_PEPPER_FRACTION = 0.03
JITTER_BRIGHTNESS_RANGE = (0.76, 1.15)
JITTER_CONTRAST_RANGE = (0.67, 1.31)
JITTER_SATURATION_RANGE = (0.68, 1.31)
JITTER_HUE_RANGE = (-0.01, 0.01)

# Experiment 2 (retired): defaults of RandomGaussianNoise, now unused.
NOISE_PROBABILITY = 0.25
NOISE_STD_RANGE = (0.02, 0.10)

Sample = tuple[str, int]
Manipulation = Callable[[Image.Image, str], Image.Image]


# ── frozen split ──────────────────────────────────────────────────────────────

def _resolve_train_root() -> Path:
    """Return the raw training folder, supporting train_set/ and train/."""
    candidates = (DATA_ROOT / "train_set", DATA_ROOT / "train")

    train_root = next((path for path in candidates if path.is_dir()), None)

    if train_root is None:
        expected = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(
            f"Training folder not found. Expected one of:\n{expected}\n"
            f"Set BIRD_DATA_ROOT to the folder holding train/ if the images "
            f"live outside the project."
        )

    return train_root


def _resolve_class_dir(train_root: Path, class_name: str, local_idx: int) -> Path:
    """Return the folder for one class, supporting 'rooster' and '00_rooster'."""
    possible_dirs = [
        train_root / class_name,
        train_root / f"{local_idx:02d}_{class_name}",
    ]

    class_dir = next((path for path in possible_dirs if path.is_dir()), None)

    if class_dir is None:
        expected = "\n".join(f"  - {path}" for path in possible_dirs)
        raise FileNotFoundError(
            f"Class folder not found for class {local_idx}: {class_name}\n"
            f"Expected one of:\n{expected}"
        )

    return class_dir


def build_split(
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
) -> tuple[list[Sample], list[Sample]]:
    """Build a stratified split, one class at a time.

    Each class is shuffled and split on its own, so every class contributes the
    same train/val ratio: 800 train / 200 val out of its 1000 images. Paths are
    sorted before the shuffle so the result does not depend on filesystem
    ordering, and the shuffle uses a local Random instance rather than the
    global random module.
    """
    train_root = _resolve_train_root()

    train_samples: list[Sample] = []
    val_samples: list[Sample] = []

    for hf_idx in sorted(TARGET_HF_INDICES):
        class_name = HF_INDEX_TO_NAME[hf_idx]
        local_idx = HF_INDEX_TO_IDX[hf_idx]

        class_dir = _resolve_class_dir(train_root, class_name, local_idx)

        image_paths: list[Path] = []
        for pattern in IMAGE_PATTERNS:
            image_paths.extend(class_dir.glob(pattern))

        if not image_paths:
            raise FileNotFoundError(
                f"No images found for class {local_idx}: {class_name}\n"
                f"Looked in {class_dir} for {', '.join(IMAGE_PATTERNS)}"
            )

        image_paths.sort()
        rng = random.Random(seed)
        rng.shuffle(image_paths)

        n_val = int(round(len(image_paths) * val_fraction))

        for position, img_path in enumerate(image_paths):
            relative_path = img_path.relative_to(DATA_ROOT).as_posix()
            sample = (relative_path, local_idx)

            if position < n_val:
                val_samples.append(sample)
            else:
                train_samples.append(sample)

    return train_samples, val_samples


def save_split(
    train_samples: list[Sample],
    val_samples: list[Sample],
    split_path: Path = SPLIT_PATH,
) -> Path:
    """Write the split to split.json and return its path."""
    split_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "train": [[path, label] for path, label in train_samples],
        "val": [[path, label] for path, label in val_samples],
    }

    split_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return split_path


def load_split(
    split_path: Path = SPLIT_PATH,
) -> tuple[list[Sample], list[Sample]]:
    """Return (train_samples, val_samples), building the split on first use."""
    if not split_path.exists():
        train_samples, val_samples = build_split()
        save_split(train_samples, val_samples, split_path)
        return train_samples, val_samples

    payload = json.loads(split_path.read_text(encoding="utf-8"))

    train_samples = [(path, int(label)) for path, label in payload["train"]]
    val_samples = [(path, int(label)) for path, label in payload["val"]]

    return train_samples, val_samples


# ── dataset ───────────────────────────────────────────────────────────────────

class BirdDataset(Dataset):
    """Reads images listed in the frozen split.

    ``pre_transform`` is the geometric part of the val pipeline, so a
    ``manipulation`` sees the final 224x224 crop; ``transform`` then turns the
    result into a normalized tensor.
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

        with Image.open(DATA_ROOT / relative_path) as image:
            image = image.convert("RGB")

        if self.pre_transform is not None:
            image = self.pre_transform(image)

        if self.manipulation is not None:
            image = self.manipulation(image, relative_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ── transforms ────────────────────────────────────────────────────────────────

class RandomGaussianNoise:
    """With probability ``p``, add N(0, sigma) noise to every pixel.

    Retired: experiment 2 wired this into train_transform_robust and it was
    dropped in experiment 3. Kept here as a record of what was tried.

    Runs on the already-normalized tensor, so ``sigma`` is in normalized units
    and is redrawn per image from ``std_range``. This is Gaussian noise only —
    salt and pepper is never applied to training data.
    """

    def __init__(
        self,
        p: float = NOISE_PROBABILITY,
        std_range: tuple[float, float] = NOISE_STD_RANGE,
    ):
        self.p = p
        self.std_range = std_range

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return tensor

        low, high = self.std_range
        std = torch.empty(1).uniform_(low, high).item()

        return tensor + torch.randn_like(tensor) * std

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p}, std_range={self.std_range})"


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Experiment 1 settings, kept after experiments 2 and 3 both lost ground on the
# held-out salt-and-pepper probe. Salt and pepper is deliberately absent — it
# stays held out for the stress set only.
train_transform_robust = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p=0.3),
    transforms.RandomApply(
        [
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.05,
            )
        ],
        p=0.5,
    ),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Must match the grader's evaluation transform exactly.
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


# ── stress-test manipulations (validation only) ───────────────────────────────

def _seed_from_path(path: str, namespace: str = "") -> int:
    """Stable 64-bit seed for a file path, identical across runs and machines."""
    digest = hashlib.sha256(f"{namespace}:{path}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def to_black_white(image: Image.Image, path: str = "") -> Image.Image:
    """Grayscale, expanded back to 3 identical RGB channels."""
    return TF.to_grayscale(image, num_output_channels=3)


def to_color_jitter(image: Image.Image, path: str = "") -> Image.Image:
    """Brightness / contrast / saturation / hue shift, fixed per image.

    Factors are drawn from ranges measured on the provided augmentations/
    examples, seeded by the file path so each image keeps its own shift.
    """
    rng = np.random.default_rng(_seed_from_path(path, "color_jitter"))

    brightness = rng.uniform(*JITTER_BRIGHTNESS_RANGE)
    contrast = rng.uniform(*JITTER_CONTRAST_RANGE)
    saturation = rng.uniform(*JITTER_SATURATION_RANGE)
    hue = rng.uniform(*JITTER_HUE_RANGE)

    image = TF.adjust_brightness(image, brightness)
    image = TF.adjust_contrast(image, contrast)
    image = TF.adjust_saturation(image, saturation)
    image = TF.adjust_hue(image, hue)
    return image


def to_salt_pepper(
    image: Image.Image,
    path: str = "",
    fraction: float = SALT_PEPPER_FRACTION,
) -> Image.Image:
    """Set ``fraction``/2 of pixels to pure black and as many to pure white."""
    array = np.array(image.convert("RGB"))
    height, width = array.shape[:2]

    n_pixels = height * width
    n_each = int(round(n_pixels * fraction / 2))

    if n_each > 0:
        rng = np.random.default_rng(_seed_from_path(path, "salt_pepper"))
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
    robust: bool = False,
) -> DataLoader:
    """Shuffled loader over the 16000 training images."""
    train_samples, _ = load_split()

    transform = train_transform_robust if robust else train_transform
    dataset = BirdDataset(train_samples, transform=transform)

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


# ── training ──────────────────────────────────────────────────────────────────

def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    """Top-1 accuracy over a loader."""
    model.eval()

    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images).argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    """Run one pass over the training set, returning the average loss."""
    model.train()

    running_loss = 0.0
    n_batches = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / n_batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ModelArchitecture from scratch.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"number of epochs, overrides EPOCHS (default: {EPOCHS})",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="train with grayscale + color-jitter augmentation (experiment 1)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=OUTPUT_NAME,
        help=(
            f"where to write the weights; a bare name lands next to train.py "
            f"(default: {OUTPUT_NAME})"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    epochs = args.epochs

    # A bare filename stays next to train.py; a path is honoured as given.
    output_path = Path(args.out).expanduser()
    if not output_path.is_absolute():
        output_path = OUTPUT_DIR / output_path

    device = pick_device()
    print(f"Device: {device}")
    print(f"Epochs: {epochs} | batch {BATCH_SIZE} | lr {LR} | wd {WEIGHT_DECAY}")

    transform_name = (
        "train_transform_robust (grayscale + color jitter)"
        if args.robust
        else "train_transform (baseline, no manipulation)"
    )
    print(f"Train transform: {transform_name}")
    print(f"Weights out:     {output_path.name}\n")

    train_loader = get_train_loader(BATCH_SIZE, NUM_WORKERS, robust=args.robust)
    val_loader = get_val_loader(BATCH_SIZE, NUM_WORKERS)

    model = ModelArchitecture(num_classes=20).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_accuracy = 0.0

    for epoch in range(1, epochs + 1):
        started = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_accuracy = evaluate(model, val_loader, device)

        scheduler.step()
        elapsed = time.time() - started

        # Save on best clean accuracy only — never just because it is the
        # last epoch.
        improved = val_accuracy > best_accuracy
        if improved:
            best_accuracy = val_accuracy

            state_dict = {
                key: value.cpu()
                for key, value in model.state_dict().items()
            }
            joblib.dump(state_dict, output_path)

        print(
            f"epoch {epoch:>3}/{epochs}  "
            f"loss {train_loss:.4f}  "
            f"val_acc {val_accuracy:.4f}  "
            f"{elapsed:6.1f}s"
            f"{'  <- saved' if improved else ''}"
        )

    if not output_path.exists():
        raise RuntimeError(f"No weights were saved to {output_path}")

    print(f"\nBest clean val accuracy: {best_accuracy:.4f}")
    print(f"Reloading {output_path.name} for the final report...\n")

    model.load_state_dict(joblib.load(output_path))
    model.to(device)

    clean_accuracy = evaluate(model, val_loader, device)

    stress_accuracies = {}
    for name in MANIPULATIONS:
        stress_loader = get_stress_loader(name, BATCH_SIZE, NUM_WORKERS)
        stress_accuracies[name] = evaluate(model, stress_loader, device)

    mean_stress = sum(stress_accuracies.values()) / len(stress_accuracies)

    print(f"{'set':<16} {'accuracy':>9}")
    print("-" * 26)
    print(f"{'clean':<16} {clean_accuracy:>9.4f}")
    for name, accuracy in stress_accuracies.items():
        print(f"{name:<16} {accuracy:>9.4f}")
    print("-" * 26)
    print(f"{'stress mean':<16} {mean_stress:>9.4f}")


if __name__ == "__main__":
    main()
