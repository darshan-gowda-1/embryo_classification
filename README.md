# 🧬 Embryo Development Stage Classification

Multi-class deep learning classification across **15 embryo development phases** using transfer learning on time-lapse microscopy images.

---

## 📋 Project Overview

This project trains and evaluates four CNN architectures to automatically classify embryo images into 15 distinct developmental stages — from the second polar body extrusion (`tPB2`) all the way through to the expanded blastocyst (`tEB`). The pipeline uses a two-stage transfer learning strategy with a custom hybrid loss function.

**Dataset:** [abhishekbuddiga06/embryo-dataset](https://www.kaggle.com/datasets/abhishekbuddiga06/embryo-dataset)  
**Framework:** PyTorch  
**Hardware:** NVIDIA Tesla T4 GPU (Google Colab)

---

## 🗂️ Dataset

| Property | Value |
|---|---|
| Total embryos | 704 |
| Total samples (frames) | 33,049 |
| Number of classes | 15 |
| Annotation format | CSV per embryo (phase, start frame, end frame) |
| Frame sampling rate | Every 10th frame from annotation range |

### Class Distribution

| Label | Phase | Samples |
|---|---|---|
| 0 | tPB2 | 1,129 |
| 1 | tPNa | 4,560 |
| 2 | tPNf | 851 |
| 3 | t2 | 3,181 |
| 4 | t3 | 754 |
| 5 | t4 | 3,160 |
| 6 | t5 | 1,073 |
| 7 | t6 | 1,098 |
| 8 | t7 | 1,284 |
| 9 | t8 | 3,475 |
| 10 | t9+ | 5,330 |
| 11 | tM | 1,921 |
| 12 | tSB | 1,900 |
| 13 | tB | 1,256 |
| 14 | tEB | 2,077 |

> The dataset is notably imbalanced — `t9+` has ~7× more samples than `t3`.

---

## 🔀 Data Split Strategy

Splits are done at the **embryo level** (not frame level) to prevent data leakage:

| Split | Samples | Embryos |
|---|---|---|
| Train | 23,078 | 492 |
| Validation | 4,962 | 106 |
| Test | 5,009 | 106 |

---

## 🏗️ Model Architectures

Four pretrained ImageNet models are evaluated:

| Model | Total Params | Notes |
|---|---|---|
| MobileNetV2 | 2.2M | Lightweight, depthwise separable convolutions |
| InceptionV3 | 24.4M | Multi-scale receptive fields, auxiliary head, 299×299 input |
| VGG16 | 134.3M | Deep uniform architecture |
| VGG19 | 139.6M | Deeper variant of VGG16 |

---

## 🔁 Two-Stage Training Pipeline

### Stage 1 — Frozen Backbone
- Backbone weights frozen; only the classifier head is trained
- Loss: `CrossEntropyLoss` with class weights
- Learning rate: `1e-3`
- Epochs: 5
- Optimizer: Adam with StepLR scheduler (step=3, γ=0.5)

### Stage 2 — Fine-Tuning (Unfrozen)
- All backbone weights are unfrozen
- Loss: `HybridLoss` (see below)
- Learning rate: `1e-4`
- Epochs: 5

---

## ⚙️ Custom Hybrid Loss Function

```
HybridLoss = 0.5 × CrossEntropy + 0.3 × FocalLoss + 0.2 × LabelSmoothing
```

| Component | Purpose |
|---|---|
| CrossEntropy (α=0.5) | Stable gradient signal throughout training |
| Focal Loss (β=0.3) | Forces focus on hard/misclassified samples and minority phases |
| Label Smoothing (γ=0.2) | Prevents over-confident predictions on ambiguous stage boundaries |

Class weights are computed from the training distribution to further address imbalance (min weight: 0.40, max weight: 2.97).

---

## 🖼️ Data Augmentation

**Training transforms:**
- Resize to model input size (224×224 or 299×299 for InceptionV3)
- Random horizontal & vertical flip
- Random rotation (±15°)
- Color jitter (brightness, contrast ±0.2)
- Random affine scaling (0.85–1.15×)
- ImageNet normalization

**Validation/Test transforms:** Resize + ImageNet normalization only.

---

## 📊 Results

### Final Test Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **InceptionV3** | **32.52%** | **37.15%** | **32.52%** | **32.69%** |
| VGG19 | 31.68% | 32.26% | 31.68% | 30.74% |
| MobileNetV2 | 30.60% | 35.73% | 30.60% | 31.30% |
| VGG16 | 14.15% | 2.00% | 14.15% | 3.51% |

**🏆 Best Model: InceptionV3** — Accuracy: 32.52%, F1: 32.69%

### MobileNetV2 — Per-Class Report (Test Set)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| tPB2 | 0.18 | 0.58 | 0.27 | 168 |
| tPNa | 0.57 | 0.33 | 0.42 | 687 |
| tPNf | 0.14 | 0.38 | 0.21 | 133 |
| t2 | 0.38 | 0.23 | 0.29 | 482 |
| t3 | 0.10 | 0.19 | 0.13 | 107 |
| t4 | 0.34 | 0.31 | 0.33 | 483 |
| t5 | 0.11 | 0.13 | 0.12 | 142 |
| t6 | 0.04 | 0.03 | 0.03 | 188 |
| t7 | 0.17 | 0.16 | 0.16 | 223 |
| t8 | 0.34 | 0.30 | 0.32 | 542 |
| t9+ | 0.51 | 0.35 | 0.42 | 709 |
| tM | 0.38 | 0.36 | 0.37 | 311 |
| tSB | 0.41 | 0.23 | 0.29 | 307 |
| tB | 0.27 | 0.22 | 0.24 | 223 |
| tEB | 0.27 | 0.58 | — | — |

### Training Curves Summary

All models show the expected pattern: Stage 1 (frozen backbone) yields limited improvement, while Stage 2 (fine-tuning) provides the main accuracy gains. VGG16 remained largely stuck due to the dominance of pretrained features and insufficient fine-tuning epochs.

---

## 🔍 Analysis & Trade-offs

### Why InceptionV3 Performs Best
- Multi-scale inception modules capture embryo features at different spatial scales
- Auxiliary classification head helps gradient flow across 15 classes
- 299×299 input resolution preserves finer morphological detail

### Why VGG16 Underperformed
- With 134M parameters, the model required more epochs to adapt from ImageNet to the embryo domain
- The Stage 2 loss barely decreased — likely needs a lower learning rate or more epochs
- F1 of only 3.51% indicates the model defaulted to predicting a narrow set of majority classes

### Speed vs Accuracy Trade-off

```
Fastest  ──── MobileNet → InceptionV3 → VGG16 → VGG19 ──── Most params
Lightest ──── MobileNet → InceptionV3 → VGG16 → VGG19 ──── Heaviest
```

- **MobileNetV2**: Best for real-time or edge deployment scenarios
- **InceptionV3**: Best balance of accuracy and model size
- **VGG16/VGG19**: Better suited for larger datasets with more training epochs

### Limitation Note
All models achieved relatively modest accuracy (~30%) on this 15-class problem. This reflects the inherent difficulty of embryo stage classification — many adjacent developmental phases (e.g. t3→t4, tSB→tB) are visually subtle, and the class imbalance amplifies errors on minority stages. Longer fine-tuning, more aggressive augmentation, or ensemble methods would likely improve performance significantly.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install kagglehub torchvision scikit-learn tqdm matplotlib seaborn
```

### 2. Download Dataset
```python
import kagglehub
path = kagglehub.dataset_download('abhishekbuddiga06/embryo-dataset')
```

### 3. Run the Notebook
Open `embryo_classification_colab.ipynb` in Google Colab and run all cells sequentially. A GPU runtime is strongly recommended.

---

## 📁 Project Structure

```
embryo_classification_colab.ipynb   # Main notebook
mobilenet_best.pth                  # Saved MobileNetV2 weights
inceptionv3_best.pth                # Saved InceptionV3 weights
vgg16_best.pth                      # Saved VGG16 weights
vgg19_best.pth                      # Saved VGG19 weights
all_models_accuracy.png             # Accuracy curves for all models
model_comparison_bar.png            # Bar chart comparing final performance
```

---

## ⚙️ Hyperparameters

| Parameter | Value |
|---|---|
| Image size | 224×224 (299×299 for InceptionV3) |
| Batch size | 8 |
| Stage 1 LR | 1e-3 |
| Stage 2 LR | 1e-4 |
| Stage 1 epochs | 5 |
| Stage 2 epochs | 5 |
| Frame sample rate | Every 10th frame |
| Random seed | 42 |

---

## 📦 Dependencies

- Python 3
- PyTorch + TorchVision
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- tqdm
- Pillow
- kagglehub

---

## 📄 License

Dataset sourced from Kaggle: [embryo-dataset](https://www.kaggle.com/datasets/abhishekbuddiga06/embryo-dataset). Please refer to the dataset page for usage terms.
