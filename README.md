# Fetal Ultrasound Binary Classification

A unified PyTorch pipeline for **binary classification of fetal ultrasound images into malignant and non-malignant classes** using deep learning, explainability, cross-validation, hyperparameter optimization, and statistical comparison.

This repository contains the notebook **`FETAL_ULTRASOUND_DATASET_ANALYSIS.ipynb`**, which consolidates data preparation, preprocessing, exploratory data analysis, training, evaluation, explainability, and supplementary classical baselines into one workflow.

## Overview

Fetal ultrasound classification is challenging because ultrasound images are noisy, low-contrast, operator-dependent, and visually heterogeneous. This project evaluates whether deep learning models can learn discriminative patterns for **malignant vs non-malignant fetal ultrasound classification** on a small, imbalanced dataset.

The notebook includes:

- dataset download and extraction
- dataset cleaning and relabeling
- leakage checks
- exploratory data analysis (EDA)
- preprocessing preserved from the original dataset notebook
- training and evaluation of multiple deep learning models
- explainability with Grad-CAM / attention rollout / Integrated Gradients
- handcrafted-feature baselines
- statistical comparison across models

## Dataset

This project uses the **Ultrasound Fetus Dataset** (Anitha, A (2026)) from Mendeley: [https://data.mendeley.com/datasets/yrzzw9m6kk/2](https://data.mendeley.com/datasets/yrzzw9m6kk/2).

Source dataset classes include:

- `normal`
- `benign`
- `malignant`

For this project, the task is converted into **binary classification**:

- `malignant`
- `non_malignant` (renamed from `benign`)

The `normal` class is removed so the task focuses on distinguishing malignant from non-malignant abnormal cases.

## What the notebook does

### 1. Data preparation

The notebook automatically:

- downloads and extracts the dataset archive if needed
- removes mask and annotation images from the classification set
- audits duplicates using MD5 hashes
- deletes the `normal` class
- renames `benign` to `non_malignant`
- creates `train / validation / test` splits
- checks for leakage across splits

### 2. Preprocessing

The notebook preserves the core preprocessing logic from the original dataset notebook through `sae_preprocess`, which applies:

1. CLAHE on the LAB luminance channel
2. Otsu thresholding
3. morphological closing
4. foreground masking

This preprocessing is wrapped in a torchvision-compatible transform and used consistently across models.

### 3. Exploratory data analysis

EDA includes:

- class distribution per split
- image size and aspect ratio distributions
- intensity statistics
- sample grids for each class
- preprocessing visualization
- GLCM texture features
- PCA visualization of handcrafted features

### 4. Deep learning models

Implemented / supported models:

- `CustomCNN`
- `ResNet50`
- `ResNet101`
- `EfficientNetB0`
- `EfficientNetB3`
- `ViT-S/16`
- `ViT-B/16` (supported in `build_model`, not always included in `MODELS_TO_RUN`)

### 5. Training pipeline

All models share a unified training pipeline with:

- transfer learning or full training depending on architecture
- phased unfreezing
- class weighting / focal loss / balanced sampling
- mixed precision training
- cosine learning-rate schedule with warmup
- gradient clipping
- early stopping on validation AUC
- optional test-time augmentation (TTA)
- optional multi-seed averaging

### 6. Evaluation

The notebook evaluates models using:

- ROC-AUC
- Precision-Recall AUC
- confusion matrix
- sensitivity / specificity
- balanced accuracy
- F1 (positive, macro, weighted)
- MCC
- Cohen's kappa
- Brier score
- calibration curves
- bootstrap confidence intervals
- threshold selection on the validation set

### 7. Explainability

Explainability methods include:

- Grad-CAM for CNN / ResNet / EfficientNet
- attention rollout for ViT
- Integrated Gradients for all models

The notebook saves examples for:

- true positives
- true negatives
- false positives
- false negatives

### 8. Supplementary classical baselines

The notebook also extracts handcrafted GLCM and first-order features and evaluates classical models such as:

- Logistic Regression
- SVM (RBF)
- Random Forest
- XGBoost (if available)

### 9. Statistical comparison

Pairwise comparison utilities include:

- DeLong test for ROC-AUC
- McNemar test for paired predictions
- Wilcoxon signed-rank test across CV folds
- Holm correction for multiple comparisons

## Repository structure

Typical outputs are saved under:

```text
results_unified/
├── eda/
├── checkpoints/
├── predictions/
├── xai/
├── metrics/
├── stats/
├── hpo/
└── ensemble/
```

Generated files include:

- EDA plots and CSV summaries
- per-model checkpoints
- saved validation/test probabilities
- training history plots
- ROC / PR / calibration plots
- XAI visualizations
- CV results
- statistical comparison tables

## Installation

Install the main dependencies used by the notebook:

```bash
pip install "timm==1.0.11" "optuna>=3.6" "grad-cam>=1.5.4" "captum>=0.7.0" \
            "scikit-image>=0.22" "umap-learn>=0.5.6" "statsmodels>=0.14" \
            "xgboost>=2.0" "seaborn>=0.13"
```

## Usage

Open and run:

```bash
FETAL_ULTRASOUND_DATASET_ANALYSIS.ipynb
```

Recommended environment:

- Google Colab or a GPU-enabled Python environment
- CUDA-capable GPU for practical training time

### Main configuration options

Important configuration variables in the notebook include:

- `IMG_SIZE`
- `MODELS_TO_RUN`
- `HPO_ENABLED`
- `HPO_TRIALS`
- `CV_FOLDS`
- `MULTI_SEED_N`
- `TTA_ENABLED`
- `THRESHOLD_STRATEGY`
- `MIN_SENSITIVITY`
- `MIN_SPECIFICITY`

## Current limitations / notes

- The dataset is small and imbalanced, so performance can be unstable across seeds and splits.
- The leading numeric token in filenames is **not globally unique across classes**, so grouping is handled using **class-qualified group identifiers**.
- If `IMG_SIZE` is set to `320`, Vision Transformer variants created as `*_224` models will fail unless image size handling is made model-specific.
- Current results are best interpreted as a strong experimental benchmark pipeline rather than a clinically deployable system.

## Recommended next steps

Based on the current notebook experiments, the most promising improvements are:

- focus on `ResNet50` as the main baseline
- add ROI / foreground cropping after preprocessing
- test higher input resolution for CNN-based models
- use gentler ultrasound-appropriate augmentation
- increase multi-seed training for more stable estimates
- run focused HPO on the best-performing backbone only

## Disclaimer

This repository is for research and educational use. It is **not** a clinical decision support system and should not be used for medical diagnosis.

## References

Anitha, A (2026), “Ultrasound Fetus Dataset”, Mendeley Data, V2, doi: 10.17632/yrzzw9m6kk.2
