from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import yaml

from src.datasets.segmentation_dataset import SegmentationDataset
from src.models.unet import UNet
from src.utils.run_logging import append_log_line, create_epoch_log, create_run_directory, save_config_snapshot


def evaluate_segmentation(config_path: str = "config.yaml") -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    manifest_path = config["data"]["manifest_path"]
    image_size = config["data"]["image_size"]
    checkpoints_dir = config["outputs"]["checkpoints_dir"]
    outputs_cfg = config.get("outputs", {})
    base_runs_dir = outputs_cfg.get("runs_dir", os.path.join("outputs", "runs"))

    run_meta = create_run_directory(base_runs_dir, config_path, run_name="segmentation_eval")
    run_dir = run_meta["run_dir"]
    save_config_snapshot(run_dir, config)
    log_path = create_epoch_log(run_dir, "evaluation_log.txt")
    append_log_line(log_path, f"Evaluation started at {run_meta['timestamp']}")
    append_log_line(log_path, f"Config copy: {run_meta['config_copy']}")

    test_ds = SegmentationDataset(manifest_path, split="test", image_size=image_size, train=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(base_channels=config["model"]["segmentation"]["base_channels"]).to(device)
    state = torch.load(os.path.join(checkpoints_dir, "segmentation_best.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()

    ious = []
    dices = []
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            preds = model(images)
            pred_bin = (preds > 0.5).float()
            inter = (pred_bin * masks).sum()
            union = pred_bin.sum() + masks.sum() - inter
            iou = inter / (union + 1e-6)
            dice = 2 * inter / (pred_bin.sum() + masks.sum() + 1e-6)
            ious.append(iou.item())
            dices.append(dice.item())

    mean_iou = float(np.mean(ious))
    mean_dice = float(np.mean(dices))
    print("Mean IoU:", mean_iou)
    print("Mean Dice:", mean_dice)
    append_log_line(log_path, f"Mean IoU: {mean_iou:.4f}")
    append_log_line(log_path, f"Mean Dice: {mean_dice:.4f}")


if __name__ == "__main__":
    evaluate_segmentation()
