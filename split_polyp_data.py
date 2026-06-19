from __future__ import annotations

import random
import shutil
from pathlib import Path


OUTPUT_ROOT = Path("polyp_data")
SEED = 42
MANAGED_SPLITS = ("train", "val", "test", "kvasir_test", "CVC-ColonDB_test", "CVC-300_test", "ETIS_test")


DATASETS = {
    "Kvasir-SEG": {
        "root": Path("Kvasir-SEG"),
        "image_dir": Path("images"),
        "mask_dir": Path("masks"),
    },
    "CvC-ClinicDB": {
        "root": Path("CvC-ClinicDB"),
        "image_dir": Path("PNG") / "Original",
        "mask_dir": Path("PNG") / "Ground Truth",
    },
    "CVC-ColonDB": {
        "root": Path("CVC-ColonDB"),
        "image_dir": Path("images"),
        "mask_dir": Path("masks"),
    },
    "CVC-300": {
        "root": Path("CVC-300"),
        "image_dir": Path("images"),
        "mask_dir": Path("masks"),
    },
    "ETIS": {
        "root": Path("ETIS"),
        "image_dir": Path("images"),
        "mask_dir": Path("masks"),
    },
}


def natural_key(path: Path) -> tuple:
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem), path.suffix.lower())
    return (1, stem.lower(), path.suffix.lower())


def collect_pairs(dataset_name: str) -> list[tuple[Path, Path]]:
    config = DATASETS[dataset_name]
    image_dir = config["root"] / config["image_dir"]
    mask_dir = config["root"] / config["mask_dir"]
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected image/mask folders for {dataset_name}: {image_dir}, {mask_dir}")

    image_paths = sorted((p for p in image_dir.iterdir() if p.is_file()), key=natural_key)
    pairs = [(p, mask_dir / p.name) for p in image_paths]
    missing = [mask for _, mask in pairs if not mask.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} masks in {dataset_name}. First missing: {missing[0]}"
        )
    return pairs


def take_split(
    pairs: list[tuple[Path, Path]],
    n_train: int,
    n_val: int,
    n_test: int,
    dataset_name: str,
) -> dict[str, list[tuple[str, Path, Path]]]:
    total_needed = n_train + n_val + n_test
    if len(pairs) < total_needed:
        raise ValueError(f"{dataset_name} has {len(pairs)} samples, need {total_needed}.")

    return {
        "train": [(dataset_name, image, mask) for image, mask in pairs[:n_train]],
        "val": [(dataset_name, image, mask) for image, mask in pairs[n_train:n_train + n_val]],
        "test": [
            (dataset_name, image, mask)
            for image, mask in pairs[n_train + n_val:n_train + n_val + n_test]
        ],
    }


def copy_split(output_root: Path, split_items: dict[str, list[tuple[str, Path, Path]]]) -> None:
    for split in MANAGED_SPLITS:
        split_root = output_root / split
        if split_root.exists():
            shutil.rmtree(split_root)

    for split, items in split_items.items():
        split_root = output_root / split
        out_images = split_root / "images"
        out_masks = split_root / "masks"

        out_images.mkdir(parents=True, exist_ok=True)
        out_masks.mkdir(parents=True, exist_ok=True)

        counts: dict[str, int] = {}
        for dataset_name, image_path, mask_path in items:
            counts[dataset_name] = counts.get(dataset_name, 0) + 1
            out_name = f"{dataset_name}_{counts[dataset_name]:04d}{image_path.suffix.lower()}"
            shutil.copy2(image_path, out_images / out_name)
            shutil.copy2(mask_path, out_masks / out_name)

        per_dataset = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        print(f"{split}: {len(items)} samples")
        print(f"  {per_dataset}")


def split_polyp_datasets(output_root: Path, seed: int = 42) -> None:
    kvasir_pairs = collect_pairs("Kvasir-SEG")
    clinic_pairs = collect_pairs("CvC-ClinicDB")
    random.Random(seed).shuffle(kvasir_pairs)
    random.Random(seed).shuffle(clinic_pairs)

    split_items = {
        "train": [],
        "val": [],
        "kvasir_test": [],
        "CVC-ColonDB_test": [],
        "CVC-300_test": [],
        "ETIS_test": [],
    }

    kvasir_split = take_split(kvasir_pairs, 700, 200, 100, "Kvasir-SEG")
    clinic_split = take_split(clinic_pairs, len(clinic_pairs), 0, 0, "CvC-ClinicDB")
    split_items["train"].extend(kvasir_split["train"])
    split_items["train"].extend(clinic_split["train"])
    split_items["val"].extend(kvasir_split["val"])
    split_items["val"].extend(clinic_split["val"])
    split_items["kvasir_test"].extend(kvasir_split["test"])

    test_splits = {
        "CVC-ColonDB_test": "CVC-ColonDB",
        "CVC-300_test": "CVC-300",
        "ETIS_test": "ETIS",
    }
    for split, dataset_name in test_splits.items():
        split_items[split].extend(
            (dataset_name, image, mask) for image, mask in collect_pairs(dataset_name)
        )

    copy_split(output_root, split_items)


def main() -> None:
    split_polyp_datasets(OUTPUT_ROOT, SEED)


if __name__ == "__main__":
    main()
