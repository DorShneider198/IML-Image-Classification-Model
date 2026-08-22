"""
Hackathon evaluation script — ImageNet subset (20 classes).

First run downloads the full ImageNet validation set to DATA_ROOT, then filters
to our 20 target classes at load time. Subsequent runs skip the download.

Prerequisites:
  pip install torch torchvision datasets Pillow
  huggingface-cli login   # one-time; requires accepting ImageNet terms at
                          # huggingface.co/datasets/imagenet-1k

Expected submissions layout:
  submissions/
    team_a/
      model.py
      weights.pt
    team_b/
      model.py
      weights.pt

Run:
  python evaluate.py
"""
import importlib.util
import sys
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from labels import (
    HF_INDEX_TO_NAME,
    HF_INDEX_TO_IDX,
    TARGET_HF_INDICES,
)

# ── editable ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_ROOT = PROJECT_ROOT / "dataset"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
BATCH_SIZE = 64
# ──────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class ImageNetSubset(Dataset):
    """Loads the 20 target classes from the selected dataset split."""

    def __init__(
        self,
        root: Path,
        split: str = "validation",
        transform=None,
    ):
        self.transform = transform
        self.samples = []

        split_root = root / split

        if not split_root.is_dir():
            raise FileNotFoundError(
                f"Dataset split folder not found: {split_root}\n"
                f"Expected structure such as:\n"
                f"  {split_root}/<class_name>/*.jpg\n"
                f"or:\n"
                f"  {split_root}/<class_index>_<class_name>/*.jpg"
            )

        for hf_idx in sorted(TARGET_HF_INDICES):
            class_name = HF_INDEX_TO_NAME[hf_idx]
            local_idx = HF_INDEX_TO_IDX[hf_idx]

            # Supports both:
            # dataset/validation/rooster
            # dataset/validation/00_rooster
            possible_dirs = [
                split_root / class_name,
                split_root / f"{local_idx:02d}_{class_name}",
            ]

            class_dir = next(
                (path for path in possible_dirs if path.is_dir()),
                None,
            )

            if class_dir is None:
                expected = "\n".join(
                    f"  - {path}" for path in possible_dirs
                )
                raise FileNotFoundError(
                    f"Class folder not found for class "
                    f"{local_idx}: {class_name}\n"
                    f"Expected one of:\n{expected}"
                )

            image_paths = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in {".jpg", ".jpeg"}
            )

            if not image_paths:
                raise FileNotFoundError(
                    f"No JPEG images found in class folder: {class_dir}"
                )

            self.samples.extend(
                (image_path, local_idx)
                for image_path in image_paths
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        with Image.open(path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

def load_test_set():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    dataset = ImageNetSubset(DATA_ROOT, split="validation", transform=transform)
    print(f"Loaded {len(dataset)} validation images across {len(TARGET_HF_INDICES)} classes.\n")

    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


# ── submission loading ────────────────────────────────────────────────────────

def load_submission(team_dir: Path):
    predict_path = team_dir / "predict.py"
    model_path = team_dir / "model.py"
    weights_path = team_dir / "weights.joblib"

    if not predict_path.exists():
        raise FileNotFoundError(f"Missing predict.py in {team_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.py in {team_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights.joblib in {team_dir}")

    original_sys_path = list(sys.path)

    old_model_module = sys.modules.pop("model", None)
    old_predict_module = sys.modules.pop("predict", None)

    try:
        # Make project root importable for base_model.py / labels.py.
        sys.path.insert(0, str(PROJECT_ROOT))

        # Make this team folder importable, so predict.py can do:
        # from model import ModelArchitecture
        sys.path.insert(0, str(team_dir))

        spec = importlib.util.spec_from_file_location(
            f"{team_dir.name}_predict",
            predict_path,
        )

        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import predict.py from {team_dir}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "Model"):
            raise AttributeError(f"predict.py in {team_dir} must define class Model")

        model = module.Model()
        model.load(str(weights_path))

        return model

    finally:
        sys.path = original_sys_path

        # Prevent one team's model.py/predict.py from leaking into the next team.
        sys.modules.pop("model", None)
        sys.modules.pop("predict", None)

        if old_model_module is not None:
            sys.modules["model"] = old_model_module
        if old_predict_module is not None:
            sys.modules["predict"] = old_predict_module

# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader):
    import torch

    correct = 0
    total = 0

    for x, y in loader:
        with torch.no_grad():
            preds = model.predict(x)

        if not isinstance(preds, torch.Tensor):
            raise TypeError("predict(x) must return a torch.Tensor")

        if preds.shape != y.shape:
            raise ValueError(
                f"predict(x) must return shape {list(y.shape)}, "
                f"got {list(preds.shape)}"
            )

        if torch.is_floating_point(preds):
            raise TypeError("predict(x) must return integer class indices, not logits/probabilities")

        preds = preds.cpu()
        y = y.cpu()

        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / total


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Preparing test set...")
    loader = load_test_set()

    team_dirs = sorted(d for d in SUBMISSIONS_DIR.iterdir() if d.is_dir())
    if not team_dirs:
        print(f"No submissions found in {SUBMISSIONS_DIR}/")
        sys.exit(1)

    results = []
    for team_dir in team_dirs:
        print(f"Evaluating {team_dir.name}...", end=" ", flush=True)
        try:
            model = load_submission(team_dir)
            acc   = evaluate(model, loader)
            results.append((team_dir.name, acc))
            print(f"accuracy: {acc:.4f}")
        except Exception as e:
            print(f"FAILED — {e}")
            results.append((team_dir.name, None))

    print("\n--- Leaderboard ---")
    ranked = sorted((r for r in results if r[1] is not None), key=lambda r: r[1], reverse=True)
    for rank, (team, acc) in enumerate(ranked, start=1):
        print(f"  {rank}. {team:<20} {acc:.4f}")
    for team, acc in results:
        if acc is None:
            print(f"  --  {team:<20} FAILED")


if __name__ == "__main__":
    main()
