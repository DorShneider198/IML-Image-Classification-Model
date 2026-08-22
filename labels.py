"""Label mapping for a 20-class ImageNet-1K bird subset.

HF_INDEX is the original 0-based ImageNet-1K label used by the
Hugging Face ILSVRC/imagenet-1k dataset.
LOCAL_INDEX is the contiguous 0-19 label used by the local model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final


HF_INDEX_TO_NAME: Final[dict[int, str]] = {
    # Distinct silhouettes / habitats
    7: "rooster",                    # HF primary label: cock
    9: "ostrich",
    22: "bald_eagle",
    23: "vulture",
    24: "great_grey_owl",
    84: "peacock",
    85: "quail",

    # Parrots: intentionally somewhat similar
    88: "macaw",
    89: "sulphur_crested_cockatoo",

    # Small / distinctive beaks
    94: "hummingbird",
    96: "toucan",

    # Waterfowl: intentionally somewhat similar
    99: "goose",
    100: "black_swan",

    # Wading birds: intentionally similar group
    127: "white_stork",
    129: "spoonbill",
    130: "flamingo",
    131: "little_blue_heron",

    # Seabirds
    144: "pelican",
    145: "king_penguin",
    146: "albatross",
}

ORDERED_HF_INDICES: Final[tuple[int, ...]] = tuple(
    sorted(HF_INDEX_TO_NAME)
)

IDX_TO_HF_INDEX: Final[dict[int, int]] = {
    local_idx: hf_idx
    for local_idx, hf_idx in enumerate(ORDERED_HF_INDICES)
}

HF_INDEX_TO_IDX: Final[dict[int, int]] = {
    hf_idx: local_idx
    for local_idx, hf_idx in IDX_TO_HF_INDEX.items()
}

IDX_TO_NAME: Final[dict[int, str]] = {
    local_idx: HF_INDEX_TO_NAME[hf_idx]
    for local_idx, hf_idx in IDX_TO_HF_INDEX.items()
}

TARGET_HF_INDICES: Final[frozenset[int]] = frozenset(
    HF_INDEX_TO_NAME
)

assert len(HF_INDEX_TO_NAME) == 20
assert set(IDX_TO_NAME) == set(range(20))


def write_labels_json(output_dir: str | Path = "dataset") -> Path:
    """Write dataset/labels.json and return its path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "labels.json"
    payload = {
        str(idx): name
        for idx, name in IDX_TO_NAME.items()
    }

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


if __name__ == "__main__":
    print(f"{'Local':<7} {'HF index':<9} {'Class'}")
    print("-" * 52)

    for local_idx, hf_idx in IDX_TO_HF_INDEX.items():
        print(
            f"{local_idx:<7} "
            f"{hf_idx:<9} "
            f"{HF_INDEX_TO_NAME[hf_idx]}"
        )

    path = write_labels_json()
    print(f"\nWrote {path}")