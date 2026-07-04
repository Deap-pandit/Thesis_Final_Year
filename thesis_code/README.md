# Automated Leaf Disease Recognition and Severity Assessment

This repository implements a dual-pipeline system for crop leaf disease recognition and severity estimation using PyTorch.

## Project Structure

- `config.yaml`: centralized paths and hyperparameters.
- `requirements.txt`: pinned dependencies.
- `data/raw/`: place the downloaded PlantVillage and Bangladeshi Crops Disease datasets here.
- `data/processed/`: preprocessed ExG-masked images and manifest.
- `src/preprocessing/`: ExG mask and dataset preparation utilities.
- `src/datasets/`: dataset classes for classification and segmentation.
- `src/models/`: DenseNet classifier and U-Net segmentation model.
- `src/train_classifier.py`, `src/train_segmentation.py`: training scripts.
- `src/evaluate_classifier.py`, `src/evaluate_segmentation.py`: evaluation scripts.
- `src/severity.py`: severity percentage calculation.
- `src/inference_pipeline.py`: end-to-end inference CLI.

## Setup

1. Create and activate a Python 3.10+ environment.
2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Download the Kaggle datasets into `data/raw/`:
   - `PlantVillage` dataset under `data/raw/PlantVillage`
   - `Bangladeshi Crops Disease Dataset` under `data/raw/Bangladeshi_Crops_Disease_Dataset`

## Run preprocessing

```bash
python src/preprocessing/prepare_dataset.py
```

This script will:
- discover classes from the source folders,
- apply ExG masking,
- resize images to 224×224,
- split data into train/val/test,
- save a manifest CSV at `data/processed/manifest.csv`.

## Notes and assumptions

- The segmentation pipeline currently lacks public lesion masks for the two datasets.
- I will add a placeholder weak-mask generator for annotation-free training as the next step.
- The class discovery and mapping are based on folder names found in the raw dataset directories.
- Default training settings are in `config.yaml`.
