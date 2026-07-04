from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import yaml

from src.datasets.classification_dataset import ClassificationDataset
from src.models.densenet_classifier import DenseNetClassifier
from src.preprocessing.prepare_dataset import PROPOSAL_CLASS_NAMES, load_manifest_class_names
from src.utils.run_logging import append_log_line, create_epoch_log, create_run_directory, save_config_snapshot


def evaluate_classifier(config_path: str = "config.yaml") -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    manifest_path = config["data"]["manifest_path"]
    image_size = config["data"]["image_size"]
    batch_size = config["training"]["classifier"]["batch_size"]
    class_names = PROPOSAL_CLASS_NAMES
    manifest_class_names = load_manifest_class_names(manifest_path)
    missing = [label for label in class_names if label not in manifest_class_names]
    if missing:
        print(f"Warning: manifest is missing {len(missing)} expected classes. Evaluation will preserve the full 19-class label ordering.")
        for label in missing:
            print(f"  - {label}")
    num_classes = len(class_names)
    config["model"]["num_classes"] = num_classes
    figures_dir = config["outputs"]["figures_dir"]
    reports_dir = config["outputs"]["reports_dir"]
    checkpoints_dir = config["outputs"]["checkpoints_dir"]
    outputs_cfg = config.get("outputs", {})
    base_runs_dir = outputs_cfg.get("runs_dir", os.path.join("outputs", "runs"))

    run_meta = create_run_directory(base_runs_dir, config_path, run_name="classifier_eval")
    run_dir = run_meta["run_dir"]
    run_figures_dir = os.path.join(run_dir, "figures")
    run_reports_dir = os.path.join(run_dir, "reports")
    os.makedirs(run_figures_dir, exist_ok=True)
    os.makedirs(run_reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    save_config_snapshot(run_dir, config)
    log_path = create_epoch_log(run_dir, "evaluation_log.txt")
    append_log_line(log_path, f"Evaluation started at {run_meta['timestamp']}")
    append_log_line(log_path, f"Config copy: {run_meta['config_copy']}")

    test_ds = ClassificationDataset(manifest_path, split="test", image_size=image_size, train=False, class_names=class_names)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNetClassifier(num_classes=num_classes, fine_tune_mode="freeze_head").to(device)
    state = torch.load(os.path.join(checkpoints_dir, "classifier_best.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()

    predictions = []
    targets = []
    probabilities = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            predictions.extend(preds.tolist())
            targets.extend(labels.tolist())
            probabilities.extend(probs.tolist())

    class_names = list(test_ds.class_to_idx.keys())
    cm = confusion_matrix(targets, predictions, labels=list(range(len(class_names))))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(run_figures_dir, "confusion_matrix.png"), dpi=300)
    plt.savefig(os.path.join(figures_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    report = classification_report(targets, predictions, target_names=class_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(run_reports_dir, "classification_report.csv"))
    report_df.to_csv(os.path.join(reports_dir, "classification_report.csv"))

    if len(class_names) > 2:
        label_binarize(targets, classes=list(range(len(class_names))))
        plt.figure(figsize=(8, 6))
        for i in range(len(class_names)):
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
            break
        plt.title("ROC Curves (One-vs-Rest)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.tight_layout()
        plt.savefig(os.path.join(run_figures_dir, "roc_curve.png"), dpi=300)
        plt.savefig(os.path.join(figures_dir, "roc_curve.png"), dpi=300)
        plt.close()

    metrics_summary = f"Evaluation complete. Saved outputs to {run_dir}"
    print(metrics_summary)
    append_log_line(log_path, metrics_summary)


if __name__ == "__main__":
    evaluate_classifier()
