# CRISPR-HyDE

**CRISPR Hybrid Deep Ensemble for Off-Target Prediction**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)


CRISPR-HyDE is a lightweight hybrid deep learning framework for predicting CRISPR-Cas9 off-target cleavage activity. It combines 3D one-hot encoded sequence representations with biologically engineered features through a dual-input architecture, aggregated via a 5-model deep ensemble optimized with Binary Focal Loss.

> **Associated Thesis:** *AI-Driven Off-Target Prediction in Gene Editing* <br>
> Humza Ahmed - MS Computer Science & Information Technology, NED University of Engineering and Technology, Karachi (Spring 2026)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
  - [Google Colab (Recommended)](#google-colab-recommended)
  - [Local Setup (Alternative)](#local-setup-alternative)
- [Usage](#usage)
  - [Arguments](#arguments)
  - [Output](#output)
- [Repository Structure](#repository-structure)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Overview

The CRISPR-Cas9 system enables precise genome editing but carries a critical risk of **off-target effects**  unintended cutting at genomic sites that share partial similarity with the intended target. CRISPR-HyDE addresses this by fusing two complementary signal sources:

1. **Sequence Branch (1D-CNN):** Processes 3D one-hot encoded sgRNA–target DNA sequence pairs (23 × 8 matrices) to capture spatial mismatch patterns.
2. **Feature Branch (Dense Network):** Processes three lightweight, biologically informative features:
   - Total mismatch count across the 23-mer alignment
   - PAM-proximal seed region mismatch count (indices 13–20)
   - GC content of the off-target sequence

A 5-model deep ensemble with distinct random seed initializations (seeds 42–46) captures diverse learned representations. The entire ensemble requires **fewer than 600,000 parameters**, making it orders of magnitude smaller than transformer-based or language-model-based alternatives.

---

## Architecture

```
                    ┌─────────────────────────┐
                    │   Input: Encoded Seqs    │
                    │     (23 × 8 matrix)      │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Conv1D (64 filters, k=3)│
                    │  BatchNorm → Dropout 0.2 │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ Conv1D (128 filters, k=3)│
                    │  MaxPool1D → Flatten     │
                    └────────────┬────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                                             │
          │                                ┌────────────▼────────────┐
          │                                │ Input: Bio Features (3) │
          │                                │   Dense (16, ReLU)      │
          │                                └────────────┬────────────┘
          │                                             │
          └──────────────────┬──────────────────────────┘
                             │ Concatenate
                ┌────────────▼────────────┐
                │    Dense (64, ReLU)      │
                │    Dropout 0.4           │
                │    Dense (1, Sigmoid)    │
                └─────────────────────────┘

              × 5 models (seeds 42–46) → Ensemble Average
```

**Per-model parameters:** ~117,697
**Total ensemble parameters:** ~588,485

---

## Key Results

Evaluated on a 15% stratified hold-out test set (22,985 sequence pairs):

| Metric | Value |
|--------|-------|
| **PR-AUC** | 0.7380 |
| **ROC-AUC** | 0.9975 |
| **Precision** | 86.67% |
| **Recall** | 39.80% |
| **F1 Score** | 0.5455 |
| **Accuracy** | 99.72% |

> **Note:** PR-AUC is the primary evaluation metric due to the extreme class imbalance (~1:230 positive-to-negative ratio). Standard accuracy is misleadingly high in such scenarios.

### Sample Output Plots

The training script automatically generates the following diagnostic plots in the `training_plots/` directory:

| Plot | Description |
|------|-------------|
| `1_pr_curve.png` | Precision-Recall curve on hold-out set |
| `2_confusion_matrix.png` | Confusion matrix at threshold 0.5 |
| `3_loss_curve.png` | Training vs. validation loss (ensemble overlay) |
| `4_pr_auc_curve.png` | Training vs. validation PR-AUC (ensemble overlay) |
| `5_roc_curve.png` | ROC curve on hold-out set |
| `6_probability_density.png` | Predicted probability density by class |

---

## Dataset

CRISPR-HyDE uses the benchmark dataset from the **DeepCRISPR** framework (Chuai et al., 2018, *Genome Biology*), obtained via the [CRISPR-DIPOFF repository](https://github.com/tzpranto/CRISPR-DIPOFF).

| Property | Value |
|----------|-------|
| Total samples | 153,233 candidate loci |
| sgRNAs | 30 (12 from K562, 18 from HEK293) |
| Positive labeling | GUIDE-seq validated cleavage sites |
| Negative labeling | Cas-OFFinder candidates without detectable activity |
| Class ratio | ~1:230 (positive : negative) |
| Mismatch tolerance | Up to 6 nucleotide mismatches |
| Format | Tab-separated: `Target sgRNA`, `Off Target sgRNA`, `label` |

The dataset file (`all_off_target.csv`) should be placed in the project root directory.

---

## Getting Started

### Google Colab (Recommended)

All training and evaluation for this research was conducted on [Google Colab](https://colab.research.google.com/) with GPU acceleration. This is the simplest way to reproduce the results with no local setup required.

1. Upload `off-target.py` and `all_off_target.csv` to your Colab session
2. Run the training script:

```python
!python off-target.py --data_path /content/all_off_target.csv --plot_dir /content/training_plots
```

3. All trained models, metrics, and diagnostic plots will be saved to the Colab session

> **Runtime:** Select **GPU** under `Runtime → Change runtime type` for significantly faster training.

### Local Setup (Alternative)

For researchers who prefer running locally:

```bash
# Clone the repository
git clone https://github.com/USERNAME/CRISPR-HyDE.git
cd CRISPR-HyDE

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Run training
python off-target.py --data_path all_off_target.csv
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `tensorflow` | ≥ 2.10 | Deep learning framework |
| `numpy` | ≥ 1.21 | Numerical computing |
| `pandas` | ≥ 1.3 | Data loading and manipulation |
| `scikit-learn` | ≥ 1.0 | Metrics, scaling, train/test split |
| `matplotlib` | ≥ 3.5 | Plotting |
| `seaborn` | ≥ 0.11 | Statistical visualizations |
| `joblib` | ≥ 1.1 | Scaler serialization |

> **Note:** All dependencies come pre-installed on Google Colab. The `requirements.txt` is only needed for local execution.

---

## Usage

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | `all_off_target.csv` | Path to the input CSV dataset |
| `--model_path` | `crispr_model` | Prefix path for saving trained model files |
| `--scaler_path` | `scaler.pkl` | Path to save the fitted StandardScaler |
| `--history_path` | `training_history` | Prefix path for saving training history CSVs |
| `--plot_dir` | `training_plots` | Directory for saving evaluation plots |
| `--num_models` | `5` | Number of ensemble models to train |

### Output

After training completes, the following files are generated:

```
├── crispr_model_0.keras          # Trained model (seed 42)
├── crispr_model_1.keras          # Trained model (seed 43)
├── crispr_model_2.keras          # Trained model (seed 44)
├── crispr_model_3.keras          # Trained model (seed 45)
├── crispr_model_4.keras          # Trained model (seed 46)
├── scaler.pkl                    # Fitted StandardScaler for feature normalization
├── training_history_0.csv        # Training metrics for model 0
├── training_history_1.csv        # Training metrics for model 1
├── ...
└── training_plots/
    ├── 1_pr_curve.png
    ├── 2_confusion_matrix.png
    ├── 3_loss_curve.png
    ├── 4_pr_auc_curve.png
    ├── 5_roc_curve.png
    └── 6_probability_density.png
```

---

## Repository Structure

```
CRISPR-HyDE/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── off-target.py                 # Main training and evaluation script
├── all_off_target.csv            # Dataset (DeepCRISPR 2018 benchmark)
└── training_plots/               # Generated evaluation plots (after training)
```

---

## Reproducibility

All random number generators are explicitly seeded to ensure deterministic results:

- **Global seed:** 42 (Python, NumPy, TensorFlow)
- **Ensemble seeds:** 42, 43, 44, 45, 46
- **TensorFlow determinism:** Enabled via `tf.config.experimental.enable_op_determinism()`
- **Train/test split:** Stratified with `random_state=42`

> **Note:** Exact numerical reproducibility requires the same hardware (GPU model), TensorFlow version, and CUDA version. Minor floating-point variations may occur across different GPU architectures.

---

## Citation



### Key References

- **DeepCRISPR (Dataset Source):** Chuai, G. et al. (2018). DeepCRISPR: optimized CRISPR guide RNA design by deep learning. *Genome Biology*, 19(1), 80.
- **CRISPR-DIPOFF (Repository Source):** Toufikuzzaman, M. et al. (2024). CRISPR-DIPOFF: an interpretable deep learning approach for CRISPR Cas-9 off-target prediction. *Briefings in Bioinformatics*, 25(2), bbad530.
- **Binary Focal Loss:** Lin, T.Y. et al. (2017). Focal loss for dense object detection. *IEEE ICCV*, 2980–2988.



## Acknowledgements

- **Supervisor:** Prof. Dr. Shariq Mahmood Khan, NED University of Engineering and Technology
- **Dataset:** DeepCRISPR benchmark dataset (Chuai et al., 2018), accessed via the CRISPR-DIPOFF repository
- **Computing:** Google Colab GPU-accelerated environment