from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import yaml

from src.models.densenet_classifier import DenseNetClassifier
from src.models.unet import UNet
from src.preprocessing.exg import apply_exg
from src.preprocessing.prepare_dataset import PROPOSAL_CLASS_NAMES, load_manifest_class_names
from src.severity import compute_severity, overlay_severity


def diagnose(image_path: str, config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masked_image, leaf_mask = apply_exg(image_rgb)
    leaf_mask = (leaf_mask > 0).astype(np.uint8)

    class_names = PROPOSAL_CLASS_NAMES
    manifest_class_names = load_manifest_class_names(config["data"]["manifest_path"])
    missing = [label for label in class_names if label not in manifest_class_names]
    if missing:
        print(f"Warning: manifest is missing {len(missing)} expected classes. Inference will still preserve the full 19-class output order.")
        for label in missing:
            print(f"  - {label}")
    num_classes = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = DenseNetClassifier(num_classes=num_classes, fine_tune_mode=config["training"]["classifier"]["fine_tune_mode"]).to(device)
    classifier.load_state_dict(torch.load(os.path.join(config["outputs"]["checkpoints_dir"], "classifier_best.pt"), map_location=device))
    classifier.eval()

    transform = torch.nn.Sequential(
        torch.nn.Resize((config["data"]["image_size"], config["data"]["image_size"])),
        torch.nn.ToTensor(),
        torch.nn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    )
    input_tensor = transform(Image.fromarray(masked_image)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = classifier(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(torch.max(probs).item())

    disease_name = class_names[pred_idx]

    segment_model = UNet(base_channels=config["model"]["segmentation"]["base_channels"]).to(device)
    segment_model.load_state_dict(torch.load(os.path.join(config["outputs"]["checkpoints_dir"], "segmentation_best.pt"), map_location=device))
    segment_model.eval()
    with torch.no_grad():
        mask_pred = segment_model(input_tensor).squeeze(0).squeeze(0).cpu().numpy()
    severity = compute_severity(mask_pred, leaf_mask)

    overlay_path = os.path.join(config["outputs"]["figures_dir"], "inference_overlay.jpg")
    overlay = overlay_severity(masked_image, mask_pred, severity, output_path=overlay_path)
    return {
        "disease": disease_name,
        "confidence": confidence,
        "severity_percent": severity,
        "overlay_image_path": overlay_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    result = diagnose(args.image, args.config)
    print(result)
