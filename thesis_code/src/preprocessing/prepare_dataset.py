import csv
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing.exg import apply_exg

PROPOSAL_CLASS_NAMES = [
    "Corn__Common_Rust",
    "Corn__Gray_Leaf_Spot",
    "Corn__Healthy",
    "Corn__Northern_Leaf_Blight",
    "Potato__Early_Blight",
    "Potato__Healthy",
    "Potato__Late_Blight",
    "Rice__Brown_Spot",
    "Rice__Healthy",
    "Rice__Leaf_Blast",
    "Rice__Neck_Blast",
    "Tomato__Early_Blight",
    "Tomato__Late_Blight",
    "Tomato__Leaf_Mold",
    "Tomato__Septoria_Leaf_Spot",
    "Tomato__Target_Spot",
    "Wheat__Brown_Rust",
    "Wheat__Healthy",
    "Wheat__Yellow_Rust",
]


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_image_file(path: Path) -> bool:
    lower_name = path.name.lower()
    return lower_name.endswith((".jpg", ".jpeg", ".png", ".bmp")) and path.is_file() and path.stat().st_size > 0


def _directory_contains_images(directory: Path) -> bool:
    return any(_is_image_file(path) for path in directory.iterdir())


def normalize_label(class_name: str) -> str:
    normalized = class_name.replace(" ", "_")
    normalized = normalized.replace("-", "_")
    normalized = normalized.replace("___", "__")
    normalized = normalized.replace("Tomato___", "Tomato__")
    normalized = normalized.replace("Tomato_", "Tomato__") if normalized.startswith("Tomato_") else normalized

    lower = normalized.lower()
    if lower.startswith("corn"):
        if "common_rust" in lower:
            return "Corn__Common_Rust"
        if "gray_leaf_spot" in lower:
            return "Corn__Gray_Leaf_Spot"
        if "healthy" in lower:
            return "Corn__Healthy"
        if "northern_leaf_blight" in lower:
            return "Corn__Northern_Leaf_Blight"

    if lower.startswith("potato"):
        if "early_blight" in lower:
            return "Potato__Early_Blight"
        if "healthy" in lower:
            return "Potato__Healthy"
        if "late_blight" in lower:
            return "Potato__Late_Blight"

    if lower.startswith("rice"):
        if "brown_spot" in lower:
            return "Rice__Brown_Spot"
        if "healthy" in lower:
            return "Rice__Healthy"
        if "leaf_blast" in lower:
            return "Rice__Leaf_Blast"
        if "neck_blast" in lower:
            return "Rice__Neck_Blast"

    if lower.startswith("wheat"):
        if "brown_rust" in lower:
            return "Wheat__Brown_Rust"
        if "healthy" in lower:
            return "Wheat__Healthy"
        if "yellow_rust" in lower:
            return "Wheat__Yellow_Rust"

    if lower.startswith("tomato"):
        if "early_blight" in lower:
            return "Tomato__Early_Blight"
        if "late_blight" in lower:
            return "Tomato__Late_Blight"
        if "leaf_mold" in lower:
            return "Tomato__Leaf_Mold"
        if "septoria" in lower:
            return "Tomato__Septoria_Leaf_Spot"
        if "target_spot" in lower:
            return "Tomato__Target_Spot"

    return normalized


def discover_classes(source_dirs: list[str]) -> set[str]:
    classes = set()
    for source_dir in source_dirs:
        source_path = Path(source_dir).resolve()
        if not source_path.exists():
            continue
        for crop_dir in source_path.iterdir():
            if not crop_dir.is_dir() or crop_dir.name.lower() in {"tomato1"}:
                continue
            for class_dir in crop_dir.iterdir():
                if class_dir.is_dir():
                    classes.add(normalize_label(class_dir.name))
    return classes


def collect_image_paths(source_dirs: list[str], class_map: dict[str, str]) -> list[tuple[str, str]]:
    samples = []
    for source_dir in source_dirs:
        source_path = Path(source_dir).resolve()
        if not source_path.exists():
            continue
        for crop_dir in source_path.iterdir():
            if not crop_dir.is_dir() or crop_dir.name.lower() in {"tomato1"}:
                continue
            for class_dir in crop_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name
                mapped_label = class_map.get(class_name)
                if mapped_label is None:
                    continue
                valid_images = []
                for image_path in sorted(class_dir.iterdir()):
                    if _is_image_file(image_path):
                        valid_images.append(image_path.resolve())
                if not valid_images:
                    continue
                for image_path in valid_images:
                    samples.append((str(image_path), mapped_label))
    return samples


def build_class_map(source_dirs: list[str]) -> dict[str, str]:
    mapping = {}
    for source_dir in source_dirs:
        source_path = Path(source_dir).resolve()
        if not source_path.exists():
            continue
        for crop_dir in source_path.iterdir():
            if not crop_dir.is_dir() or crop_dir.name.lower() in {"tomato1"}:
                continue
            for class_dir in crop_dir.iterdir():
                if class_dir.is_dir():
                    mapping[class_dir.name] = normalize_label(class_dir.name)
    return mapping


def load_manifest_class_names(manifest_path: str) -> list[str]:
    df = pd.read_csv(manifest_path)
    return sorted(df["label"].unique())


def get_ordered_class_names(manifest_path: str) -> list[str]:
    found_labels = load_manifest_class_names(manifest_path)
    extra = [label for label in found_labels if label not in PROPOSAL_CLASS_NAMES]
    return PROPOSAL_CLASS_NAMES + extra


def validate_expected_classes(found_labels: set[str]) -> tuple[list[str], list[str]]:
    expected = set(PROPOSAL_CLASS_NAMES)
    missing = sorted(expected - found_labels)
    extra = sorted(found_labels - expected)
    return missing, extra


def prepare_processed_image(image_path: str, output_path: str, image_size: int, threshold: float | None):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masked, mask = apply_exg(image, threshold=threshold)
    resized = cv2.resize(masked, (image_size, image_size), interpolation=cv2.INTER_AREA)
    output_path_parent = Path(output_path).parent
    output_path_parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
    return output_path


def generate_manifest(samples: list[tuple[str, str]], output_csv: str, processed_dir: str, split_ratios: dict[str, float], seed: int):
    labels = [label for _, label in samples]
    unique_labels = sorted(set(labels))

    df = pd.DataFrame(samples, columns=["filepath", "label"])
    df["label"] = pd.Categorical(df["label"], categories=unique_labels)

    train_val, test = train_test_split(df, test_size=split_ratios["test"], stratify=df["label"], random_state=seed)
    train, val = train_test_split(
        train_val,
        test_size=split_ratios["val"] / (split_ratios["train"] + split_ratios["val"]),
        stratify=train_val["label"],
        random_state=seed,
    )

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    manifest = pd.concat([train, val, test], ignore_index=True)
    manifest = manifest.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    manifest.to_csv(output_csv, index=False)
    return manifest


def process_dataset(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    raw_dir = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]
    manifest_path = config["data"]["manifest_path"]
    image_size = config["data"]["image_size"]
    split_ratios = config["data"]["train_val_test_split"]
    source_dirs = config["data"]["source_dirs"]
    threshold = None if config["data"]["use_otsu"] else config["data"]["fixed_threshold"]
    seed = config["data"]["random_seed"]

    discovered = discover_classes(source_dirs)
    class_map = build_class_map(source_dirs)

    print("Discovered classes:")
    for class_name in sorted(discovered):
        print(f"  - {class_name}")
    print(f"Total discovered classes: {len(discovered)}")
    print("Expected thesis classes:")
    for class_name in PROPOSAL_CLASS_NAMES:
        print(f"  - {class_name}")

    samples = collect_image_paths(source_dirs, class_map)
    if len(samples) == 0:
        raise RuntimeError("No image samples were found. Verify raw data paths in config.yaml.")

    print(f"Total image samples found: {len(samples)}")
    counts = Counter([label for _, label in samples])
    print("Sample counts by class:")
    for label, count in sorted(counts.items()):
        print(f"  - {label}: {count}")

    missing_expected, extra_classes = validate_expected_classes(set(counts))
    if missing_expected:
        print("Warning: expected classes not found or with no valid image files:")
        for missing in missing_expected:
            print(f"  - {missing}")
        print("Proceeding with the available classes for training/evaluation.")
    if extra_classes:
        print("Warning: unexpected classes were found in the dataset:")
        for extra in extra_classes:
            print(f"  - {extra}")

    if len(counts) != len(PROPOSAL_CLASS_NAMES):
        print(f"Note: {len(counts)} valid classes were discovered; expected {len(PROPOSAL_CLASS_NAMES)} proposal classes.")

    processed_manifest = []
    for image_path, label in samples:
        source_path = Path(image_path)
        if not source_path.exists() or source_path.stat().st_size == 0:
            print(f"Skipping invalid image file: {image_path}")
            continue
        relative_path = source_path.relative_to(Path(raw_dir).resolve().parent)
        output_rel_path = Path(processed_dir) / relative_path
        processed_path = str(output_rel_path)
        try:
            prepare_processed_image(image_path, processed_path, image_size, threshold)
            processed_manifest.append((processed_path, label))
        except Exception as exc:
            print(f"Warning: failed to process {image_path}: {exc}")

    if len(processed_manifest) == 0:
        raise RuntimeError("No images were successfully processed.")

    generate_manifest(processed_manifest, manifest_path, processed_dir, split_ratios, seed)
    print(f"Manifest written to {manifest_path}")
    print("Detected class labels:", sorted({row[1] for row in processed_manifest}))


if __name__ == "__main__":
    process_dataset()
