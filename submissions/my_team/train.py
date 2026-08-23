"""Baseline training run: ModelArchitecture from scratch, clean data only.

No manipulation is ever applied to the training set — the stress loaders are
used at the end for reporting only. Produces weights.joblib next to this file.

Run:
  python train.py
  python train.py --epochs 1     # quick sanity run
  python train.py --robust --out weights_robust.joblib
"""

import argparse
import sys
import time
from pathlib import Path

import joblib
import torch
import torch.nn as nn

# data.py / split_data.py sit two levels up in the repo, but next to this file
# in a flat deployment (Kaggle). Support both.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (
    SCRIPT_DIR
    if (SCRIPT_DIR / "data.py").exists()
    else SCRIPT_DIR.parents[1]
)
sys.path.insert(0, str(PROJECT_ROOT))

from data import (  # noqa: E402
    MANIPULATIONS,
    get_stress_loader,
    get_train_loader,
    get_val_loader,
)
from model import ModelArchitecture  # noqa: E402


EPOCHS = 30
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

OUTPUT_NAME = "weights.joblib"
OUTPUT_DIR = Path(__file__).resolve().parent


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
    parser = argparse.ArgumentParser(description=__doc__)
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
    print(f"Weights out:     {output_path}\n")

    train_loader = get_train_loader(BATCH_SIZE, NUM_WORKERS, robust=args.robust)
    val_loader = get_val_loader(BATCH_SIZE, NUM_WORKERS)

    model = ModelArchitecture(num_classes=20).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    best_accuracy = 0.0

    for epoch in range(1, epochs + 1):
        started = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
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
