# 🧬 Kaggle Stanford RNA 3D Structure Prediction (Silver Medal Solution)

This repository contains our silver medal-winning solution to the [Kaggle Stanford RNA 3D Structure Prediction Challenge](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/overview). The objective was to predict the 3D structure of RNA molecules with high accuracy using deep learning and biophysical modeling techniques.

> 🥈 **Silver Medal (Top 3%) | Team Score: 0.395 | Team Name: Mohammad Umam Ali**

---

## 🚀 Overview

Our pipeline builds upon the open-source DRfold2 repository, and we introduced multiple key enhancements to improve both performance and efficiency:

- **Base Model**: Forked and extended from the original [DRfold2 GitHub repository](https://github.com/leeyang/DRfold2).
- **Global RNA Language Model Loading**: Leveraged global RNALM loading for efficiency across multiple folds.
- **Energy-Based Structure Selection**:
  - Used **quadratic and linear energy functions** to filter poor-quality predictions early in the pipeline.
- **Top Structure Refinement**:
  - The best structures from the initial DRfold2 output were further refined using **OpenMM** for better local geometry.
- **Model Ensembling**:
  - Final predictions were **ensembled with Protenix** outputs to boost robustness and capture diverse structural cues.
- **Pipeline Optimization**:
  - Significant speedup achieved by removing redundant computations and cleaning the original DRfold2 workflow.

---

## 🧪 Model Architecture & Workflow
Input FASTA → RNALM Embeddings → DRfold2 Predictions → Energy Filter → OpenMM Refinement → Protenix Ensemble → Final PDB

- **DRfold2**: Deep learning-based RNA structure prediction.
- **RNALM**: Pre-trained RNA language model used for embedding extraction.
- **OpenMM**: Molecular dynamics engine for refining atomic-level structures.
- **Protenix**: External RNA prediction model used for ensembling.

---

## 📁 Folder Structure
Kaggle-Stanford-RNA-Prediction/
│
├── drfold2/ # Optimized and modified DRfold2 code
├── data/ # Preprocessed RNA data and sample inputs
├── rna_utils/ # Utilities for RNALM loading and ensembling
├── notebooks/ # EDA, testing, and visualization
├── ensemble/ # Code to ensemble DRfold2 and Protenix predictions
└── run_pipeline.py # Main entry point for end-to-end inference

---

## 📊 Performance

- LB Score: **0.407**
- Model Generalization: Competitive across both long and short RNA sequences.

---

## 📦 Requirements

- Python 3.9+
- PyTorch ≥ 1.13
- OpenMM
- Biopython
- pdbfixer
- RDKit

Install dependencies using:

```bash
pip install -r requirements.txt

🔗 References
🔬 Original DRfold2 Repository: https://github.com/leeyang/DRfold2

🧠 Protenix (used in ensembling):https://www.kaggle.com/datasets/geraseva/protenix-checkpoints

🧾 Competition Link: https://www.kaggle.com/competitions/stanford-rna-3d-folding

