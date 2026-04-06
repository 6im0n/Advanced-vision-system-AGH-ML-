<div align="center">

# 🦷 Advanced Vision System — Toothbrush Bristle Defect Detection

**AGH University of Science and Technology · Machine Learning Project**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

> *Automated quality control of toothbrush bristles using computer vision and machine learning — from raw SEM images to defect classification.*

</div>

---

## 📑 Table of Contents

1. [📖 Background](#-background)
2. [🔍 Defect Types](#-defect-types)
3. [🎯 Project Objectives](#-project-objectives)
4. [🔬 Methodology](#-methodology)
5. [🗂️ Dataset](#️-dataset)
6. [🛠️ Tech Stack](#️-tech-stack)
7. [📊 Evaluation](#-evaluation)
8. [🚀 Getting Started](#-getting-started)
9. [📁 Project Structure](#-project-structure)
10. [🤝 Contributing](#-contributing)
11. [📜 License](#-license)

---

## 📖 Background

Detection of toothbrush bristle defects is carried out mainly using **vision systems** and **machine learning**. Data for this type of task includes:

- 📷 **High-resolution images** — capturing bristle geometry and surface texture
- 🔬 **Scanning Electron Microscope (SEM) photographs** — enabling analysis of bristle tip morphology at the micro-scale

These imaging modalities make it possible to precisely analyse the **quality of bristle tips** in a manufacturing environment, enabling automated quality control that is both fast and consistent.

In production systems (quality control), the most commonly detected issues directly affect the product's cleaning efficiency and safety for the user.

---

## 🔍 Defect Types

The following bristle defects are targeted by this vision system:

| # | Defect | Description |
|---|--------|-------------|
| 1 | 🌿 **Splayed / Spread Bristles** | Deformation of bristles, causing them to fan out incorrectly |
| 2 | 🪨 **Abrasion** | Damage or wear at the bristle tips reducing cleaning effectiveness |
| 3 | 📌 **Fusing / Mounting Errors** | Incorrect seating of tufts — bristle stapling defects |
| 4 | ⭕ **Improper Rounding of Tips** | Bristle tips that are not rounded correctly, posing user safety risks |
| 5 | ❌ **Missing Bristles** | Tufts with absent or insufficient bristles |

---

## 🎯 Project Objectives

The objective of this project is to **develop and test the concept of a vision system** for automatic detection of selected toothbrush bristle defects.

The project covers the full ML pipeline:

- [x] **Data preparation** — collect images and define defect classes
- [x] **Preprocessing** — normalisation, denoising, contrast enhancement
- [x] **Region of Interest extraction** — segmentation of bristle tips or feature extraction describing the defects
- [x] **Model selection & training** — classification/detection model or rule-based detector
- [x] **Evaluation** — accuracy, precision/recall, confusion matrix, and discussion of limitations

---

## 🔬 Methodology

The system follows a structured computer vision pipeline:

```
Raw Images (High-Res / SEM)
         │
         ▼
┌─────────────────────┐
│   1. Preprocessing  │  ← Normalisation, Denoising, Contrast Enhancement
└─────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  2. Segmentation / ROI   │  ← Bristle tip localisation
└──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  3. Feature Extraction   │  ← Shape, texture, morphology descriptors
└──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  4. Classification /     │  ← ML model or rule-based detector
│     Detection Model      │
└──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  5. Evaluation & Report  │  ← Accuracy, Precision, Recall, F1, CM
└──────────────────────────┘
```

### Preprocessing Steps

| Step | Technique | Purpose |
|------|-----------|---------|
| Normalisation | Min-max / Z-score | Standardise pixel intensity |
| Denoising | Gaussian / Median filter | Remove sensor noise |
| Contrast Enhancement | CLAHE / Histogram Equalisation | Improve feature visibility |
| Resizing | Bilinear interpolation | Uniform input dimensions |

---

## 🗂️ Dataset

The dataset consists of high-resolution images and SEM photographs of toothbrush bristles, labelled by defect type.

| Property | Details |
|----------|---------|
| 📦 Format | Images (JPEG / PNG / TIFF) |
| 🏷️ Labels | 5 defect classes + normal |
| 🔬 Modalities | Standard photography & SEM |

> 📥 **Download the dataset:**
> [Google Drive — Bristle Dataset](https://drive.google.com/drive/folders/1nXdqY60uZIEWWwLzuxtzRkXygI2xNbS6?usp=sharing)

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) |
| Computer Vision | ![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?logo=opencv&logoColor=white) |
| Machine Learning | ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white) |
| Deep Learning | ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch&logoColor=white) / ![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?logo=tensorflow&logoColor=white) |
| Data Handling | ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white) |
| Visualisation | ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C) ![Seaborn](https://img.shields.io/badge/-Seaborn-4C72B0) |
| Notebooks | ![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?logo=jupyter&logoColor=white) |

---

## 📊 Evaluation

Model performance is assessed using the following metrics:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions / total samples |
| **Precision** | True positives / (true positives + false positives) per class |
| **Recall** | True positives / (true positives + false negatives) per class |
| **F1-Score** | Harmonic mean of precision and recall |
| **Confusion Matrix** | Per-class breakdown of predictions vs. ground truth |

Results and a discussion of system limitations are documented in the project report.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip / conda

### Installation

```bash
# Clone the repository
git clone https://github.com/6im0n/Advanced-vision-system-AGH-ML-.git
cd Advanced-vision-system-AGH-ML-

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

1. Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1nXdqY60uZIEWWwLzuxtzRkXygI2xNbS6?usp=sharing)
2. Place the data in a `data/` folder at the project root:

```
data/
├── train/
│   ├── normal/
│   ├── splayed/
│   ├── abrasion/
│   ├── mounting_error/
│   ├── improper_rounding/
│   └── missing_bristles/
└── test/
    └── ...
```

### Running the Pipeline

```bash
# Run the full detection pipeline
python main.py

# Or open the interactive notebook
jupyter notebook notebooks/bristle_defect_detection.ipynb
```

---

## 📁 Project Structure

```
Advanced-vision-system-AGH-ML-/
├── data/                     # Dataset (not tracked by git)
│   ├── train/
│   └── test/
├── notebooks/                # Jupyter notebooks for exploration & experiments
├── src/
│   ├── preprocessing/        # Normalisation, denoising, contrast enhancement
│   ├── segmentation/         # Bristle tip ROI extraction
│   ├── features/             # Feature extraction
│   ├── models/               # Classification / detection models
│   └── evaluation/           # Metrics, confusion matrix, visualisations
├── results/                  # Output images, metrics, model checkpoints
├── requirements.txt          # Python dependencies
├── main.py                   # Entry point
└── README.md
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ at **AGH University of Science and Technology**

*Advanced Vision Systems · Machine Learning · 2024/2025*

</div>

