"""Frozen, deterministic, stratified train/validation split of the raw training data.

This module never copies or moves images — it only records relative file paths
in dataset/split.json so every run, machine and model sees the same split.

Import ``load_split()`` from train.py to get the two lists of samples.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Final

from labels import (
    HF_INDEX_TO_NAME,
    HF_INDEX_TO_IDX,
    TARGET_HF_INDICES,
)


VAL_FRACTION: Final[float] = 0.2
SEED: Final[int] = 42

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DATASET_DIR: Final[Path] = PROJECT_ROOT / "dataset"

# The raw training images. Supports both dataset/train_set and dataset/train.
TRAIN_ROOT_CANDIDATES: Final[tuple[Path, ...]] = (
    DATASET_DIR / "train_set",
    DATASET_DIR / "train",
)

SPLIT_PATH: Final[Path] = DATASET_DIR / "split.json"

IMAGE_PATTERNS: Final[tuple[str, ...]] = ("*.jpg", "*.jpeg", "*.JPEG", "*.png")

Sample = tuple[str, int]


def _resolve_train_root() -> Path:
    """Return the raw training folder, supporting train_set/ and train/."""
    train_root = next(
        (path for path in TRAIN_ROOT_CANDIDATES if path.is_dir()),
        None,
    )

    if train_root is None:
        expected = "\n".join(f"  - {path}" for path in TRAIN_ROOT_CANDIDATES)
        raise FileNotFoundError(
            f"Training folder not found. Expected one of:\n{expected}"
        )

    return train_root


def _resolve_class_dir(train_root: Path, class_name: str, local_idx: int) -> Path:
    """Return the folder for one class, supporting 'rooster' and '00_rooster'."""
    possible_dirs = [
        train_root / class_name,
        train_root / f"{local_idx:02d}_{class_name}",
    ]

    class_dir = next(
        (path for path in possible_dirs if path.is_dir()),
        None,
    )

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
    same train/val ratio. Paths are sorted before the shuffle so the result does
    not depend on filesystem ordering.
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

        # Sort first so the order is identical on every machine, then shuffle
        # with a local Random instance (never the global random module).
        image_paths.sort()
        rng = random.Random(seed)
        rng.shuffle(image_paths)

        n_val = int(round(len(image_paths) * val_fraction))

        for position, img_path in enumerate(image_paths):
            relative_path = img_path.relative_to(PROJECT_ROOT).as_posix()
            sample = (relative_path, local_idx)

            if position < n_val:
                val_samples.append(sample)
            else:
                train_samples.append(sample)

    return train_samples, val_samples


def save_split(
    train_samples: list[Sample],
    val_samples: list[Sample],
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
    split_path: Path = SPLIT_PATH,
) -> Path:
    """Write the split to dataset/split.json and return its path."""
    split_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "seed": seed,
        "val_fraction": val_fraction,
        "train": [[path, label] for path, label in train_samples],
        "val": [[path, label] for path, label in val_samples],
    }

    split_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return split_path


def load_split(
    split_path: Path = SPLIT_PATH,
) -> tuple[list[Sample], list[Sample]]:
    """Return (train_samples, val_samples), building the split on first use."""
    if not split_path.exists():
        train_samples, val_samples = build_split(VAL_FRACTION, SEED)
        save_split(train_samples, val_samples, VAL_FRACTION, SEED, split_path)
        return train_samples, val_samples

    payload = json.loads(split_path.read_text(encoding="utf-8"))

    train_samples = [(path, int(label)) for path, label in payload["train"]]
    val_samples = [(path, int(label)) for path, label in payload["val"]]

    return train_samples, val_samples


def _count_per_class(samples: list[Sample]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for _, local_idx in samples:
        counts[local_idx] = counts.get(local_idx, 0) + 1
    return counts


if __name__ == "__main__":
    train_samples, val_samples = build_split(VAL_FRACTION, SEED)
    path = save_split(train_samples, val_samples, VAL_FRACTION, SEED)

    train_counts = _count_per_class(train_samples)
    val_counts = _count_per_class(val_samples)

    print(f"{'Idx':<5} {'Class':<28} {'Train':>7} {'Val':>7}")
    print("-" * 50)

    for hf_idx in sorted(TARGET_HF_INDICES):
        local_idx = HF_INDEX_TO_IDX[hf_idx]
        print(
            f"{local_idx:<5} "
            f"{HF_INDEX_TO_NAME[hf_idx]:<28} "
            f"{train_counts.get(local_idx, 0):>7} "
            f"{val_counts.get(local_idx, 0):>7}"
        )

    print("-" * 50)
    print(f"{'':<5} {'TOTAL':<28} {len(train_samples):>7} {len(val_samples):>7}")

    train_paths = {path for path, _ in train_samples}
    val_paths = {path for path, _ in val_samples}
    overlap = train_paths & val_paths

    assert not overlap, f"Train/val overlap: {len(overlap)} paths"
    print(f"\nOverlap between train and val: {len(overlap)} paths")
    print(f"Wrote {path}")
