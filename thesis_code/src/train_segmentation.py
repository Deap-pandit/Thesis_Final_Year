from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import yaml

from src.datasets.segmentation_dataset import SegmentationDataset
from src.models.unet import UNet
from src.utils.run_logging import append_log_line, create_epoch_log, create_run_directory, save_config_snapshot


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    smooth = 1.0
    intersection = (pred * target).sum()
    return 1.0 - ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def train_segmentation(config_path: str = "config.yaml") -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    manifest_path = config["data"]["manifest_path"]
    image_size = config["data"]["image_size"]
    batch_size = int(config["training"]["segmentation"]["batch_size"])
    epochs = int(config["training"]["segmentation"]["num_epochs"])
    lr = float(config["training"]["segmentation"]["learning_rate"])
    weight_decay = float(config["training"]["segmentation"]["weight_decay"])
    patience = int(config["training"]["segmentation"]["early_stopping_patience"])
    base_channels = int(config["model"]["segmentation"]["base_channels"])
    checkpoints_dir = config["outputs"]["checkpoints_dir"]
    outputs_cfg = config.get("outputs", {})
    base_runs_dir = outputs_cfg.get("runs_dir", os.path.join("outputs", "runs"))

    run_meta = create_run_directory(base_runs_dir, config_path, run_name="segmentation")
    run_dir = run_meta["run_dir"]
    run_checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(run_checkpoints_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    save_config_snapshot(run_dir, config)
    log_path = create_epoch_log(run_dir, "training_log.txt")
    append_log_line(log_path, f"Run started at {run_meta['timestamp']}")
    append_log_line(log_path, f"Config copy: {run_meta['config_copy']}")

    train_ds = SegmentationDataset(manifest_path, split="train", image_size=image_size, train=True)
    val_ds = SegmentationDataset(manifest_path, split="val", image_size=image_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(base_channels=base_channels).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} train", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            bce_loss = criterion(outputs, masks)
            d_loss = dice_loss(outputs, masks)
            loss = bce_loss + d_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} val", leave=False):
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                bce_loss = criterion(outputs, masks)
                d_loss = dice_loss(outputs, masks)
                val_loss += (bce_loss + d_loss).item() * images.size(0)

        val_loss = val_loss / len(val_ds)
        scheduler.step(val_loss)

        log_message = f"Epoch {epoch + 1}/{epochs} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}"
        print(log_message)
        append_log_line(log_path, log_message)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(run_checkpoints_dir, "segmentation_best.pt"))
            torch.save(best_state, os.path.join(checkpoints_dir, "segmentation_best.pt"))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered")
                append_log_line(log_path, "Early stopping triggered")
                break

    if best_state is not None:
        torch.save(best_state, os.path.join(run_checkpoints_dir, "segmentation_final.pt"))
        torch.save(best_state, os.path.join(checkpoints_dir, "segmentation_final.pt"))

    append_log_line(log_path, f"Training completed. Best validation loss: {best_val_loss:.4f}")
    print(f"Run artifacts saved to {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train_segmentation(args.config)
