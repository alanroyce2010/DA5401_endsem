
#  Score Prediction Pipeline — README

This repository contains a complete end-to-end pipeline for **score prediction** using multiple modeling strategies, extensive **data augmentation**, and detailed **exploratory data analysis (EDA)**.
The project combines traditional ML, neural networks, Gaussian Process Regression (GPR), and a Mixture-of-Experts (MoE) framework to address the multimodal nature of the dataset.

---

#  Repository Structure

```
.
├── eda.ipynb              # Exploratory data analysis
├── augmentation.ipynb     # Score + embedding data augmentation
├── ollama.ipynb           # LLM-powered augmentation via Ollama
├── nn.ipynb               # Neural network regression model
├── gpr.ipynb              # Gaussian Process Regression + bin corrections
├── MoE.ipynb              # Mixture-of-Experts model (multi-region prediction)
├── data/
│   ├── train_data.json
│   ├── test_data.json
│   ├── metric_name_embeddings.npy
│   ├── X_train_augmented.npy
│   ├── y_train_augmented.npy
│   └── other intermediate arrays
└── README.md              # This file
```

---

# Project Overview

The core task is to predict a numerical **score (0–10)** for each prompt–metric pair.
The dataset exhibits *clustered score distributions* (low, medium, high), calling for specialized models and augmentation techniques.

To address this, the project provides:

###  1. Comprehensive EDA

`eda.ipynb` explores:

* Score distributions
* Prompt/metric properties
* Embedding PCA visualizations
* Correlations and feature patterns

This informs later modeling and augmentation strategies.

---

### 🔧 2. Data Augmentation

Two kinds of augmentation are implemented:

#### **A. Statistical augmentation** (`augmentation.ipynb`)

* Noise-based score perturbation
* Synthetic re-embedding of concatenated prompts
* Region-aware augmentation to balance low/medium/high score bins

#### **B. LLM augmentation with Ollama** (`ollama.ipynb`)

* Uses a local LLM (e.g., Llama 3.2)
* Generates alternative phrasing, synthetic samples, or richer variations
* Caching ensures efficiency and reproducibility

---

###  3. Feature Construction

Across notebooks, features include:

* Sentence-BERT embeddings
* Metric embeddings
* PCA components
* Prompt length and metadata
* Augmentation-enhanced embeddings

These are stored as standardized `.npy` arrays.

---

#  Modeling Approaches

The repository implements multiple models, each suited for different dataset properties.

---

## 1. **Neural Network Regressor** (`nn.ipynb`)

A feedforward PyTorch model with:

* Linear → ReLU → Dropout blocks
* Early stopping
* Train/validation split

Works well after PCA reduction and augmentation.

---

## 2. **Gaussian Process Regression (GPR)** (`gpr.ipynb`)

A probabilistic model with:

* RBF kernel
* Predictive mean + predictive uncertainty
* Post-processing using bin-aware truncation
* Expected values under truncated normal distributions

Helps avoid extreme mispredictions caused by score clustering.

---

## 3. **Mixture-of-Experts (MoE)** (`MoE.ipynb`)

A gating-network approach where:

* Each expert specializes in a *score region*
* Gating network assigns weights dynamically
* Final output = weighted sum of expert outputs

Solves “regression-to-the-mean” and handles multimodal distributions effectively.

---

#  Outputs

The notebooks generate final predictions in:

```
submission.csv
gpr_submission.csv
moe_submission.csv
```

Based on the model executed.

---

# ▶How to Run the Pipeline

### **1. Install dependencies**

```
pip install -r requirements.txt
```

(Note: If you want, I can generate a `requirements.txt` based on your notebooks.)

### **2. Ensure Ollama is running (optional)**

```
ollama serve
ollama run llama3.2
```

### **3. Run notebooks in the recommended order:**

1. `eda.ipynb`
2. `augmentation.ipynb`
3. `ollama.ipynb` (optional LLM augmentation)
4. `nn.ipynb` / `gpr.ipynb` / `MoE.ipynb`

### **4. Collect predictions**

Each notebook saves a submission file ready for evaluation.

---

# Key Insights Learned

* The dataset's **three natural score clusters** justify bin-aware modeling.
* Augmentation (especially LLM-based) significantly improves generalization.
* PCA stabilizes high-dimensional embedding-based models.
* MoE provides the strongest modelling behaviour for multimodal score distributions.
* GPR uncertainty provides useful correction mechanisms.

---

#  Conclusion

This repository provides a full pipeline—from data exploration to augmentation to advanced model architectures—for robust score prediction.
It is modular, extensible, and designed for experimentation with embeddings, probabilistic models, and deep learning approaches.

---

