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

from src.datasets.classification_dataset import ClassificationDataset
from src.models.densenet_classifier import DenseNetClassifier
from src.preprocessing.prepare_dataset import PROPOSAL_CLASS_NAMES, load_manifest_class_names
from src.utils.run_logging import append_log_line, create_epoch_log, create_run_directory, save_config_snapshot


def train_classifier(config_path: str = "config.yaml") -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    manifest_path = config["data"]["manifest_path"]
    image_size = config["data"]["image_size"]
    batch_size = config["training"]["classifier"]["batch_size"]
    epochs = config["training"]["classifier"]["num_epochs"]
    lr = float(config["training"]["classifier"]["learning_rate"])
    weight_decay = float(config["training"]["classifier"]["weight_decay"])
    patience = config["training"]["classifier"]["early_stopping_patience"]
    fine_tune_mode = config["training"]["classifier"]["fine_tune_mode"]
    class_names = PROPOSAL_CLASS_NAMES
    manifest_labels = load_manifest_class_names(manifest_path)
    missing = [label for label in class_names if label not in manifest_labels]
    if missing:
        print(f"Warning: manifest is missing {len(missing)} expected classes. Classifier will still use {len(class_names)} output classes.")
        for label in missing:
            print(f"  - {label}")
    num_classes = len(class_names)
    config["model"]["num_classes"] = num_classes
    checkpoints_dir = config["outputs"]["checkpoints_dir"]
    outputs_cfg = config.get("outputs", {})
    base_runs_dir = outputs_cfg.get("runs_dir", os.path.join("outputs", "runs"))

    run_meta = create_run_directory(base_runs_dir, config_path, run_name="classifier")
    run_dir = run_meta["run_dir"]
    run_checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(run_checkpoints_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    save_config_snapshot(run_dir, config)
    log_path = create_epoch_log(run_dir, "training_log.txt")
    append_log_line(log_path, f"Run started at {run_meta['timestamp']}")
    append_log_line(log_path, f"Config copy: {run_meta['config_copy']}")

    train_ds = ClassificationDataset(manifest_path, split="train", image_size=image_size, train=True, class_names=class_names)
    val_ds = ClassificationDataset(manifest_path, split="val", image_size=image_size, train=False, class_names=class_names)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNetClassifier(num_classes=num_classes, fine_tune_mode=fine_tune_mode).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} train", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_ds)
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} val", leave=False):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_ds)
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        log_message = f"Epoch {epoch + 1}/{epochs} | train loss: {train_loss:.4f} | train acc: {train_acc:.4f} | val loss: {val_loss:.4f} | val acc: {val_acc:.4f}"
        print(log_message)
        append_log_line(log_path, log_message)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(run_checkpoints_dir, "classifier_best.pt"))
            torch.save(best_state, os.path.join(checkpoints_dir, "classifier_best.pt"))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered")
                append_log_line(log_path, "Early stopping triggered")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(run_checkpoints_dir, "classifier_final.pt"))
        torch.save(best_state, os.path.join(checkpoints_dir, "classifier_final.pt"))

    append_log_line(log_path, f"Training completed. Best validation loss: {best_val_loss:.4f}")
    print(f"Run artifacts saved to {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train_classifier(args.config)
