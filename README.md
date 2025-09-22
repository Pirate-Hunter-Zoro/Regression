
# Regression and Classification Project

This repository contains implementations and experiments for a machine learning project covering **Linear Regression**, **Logistic Regression**, and **Digits Classification**. The work includes hyperparameter sweeps, regularization, closed-form solutions, and visualization of results.

---

## Project Structure

```bash
├── src/
│   ├── datasets.py          # Synthetic and real dataset loaders
│   ├── models_linear.py     # Linear regression models (GD + closed-form ridge)
│   ├── models_logistic.py   # Logistic regression models
│   ├── cv.py                # Cross-validation and grid search
│   ├── utils.py             # Helpers (standardization, bias column, etc.)
│   ├── plots.py             # Simple plotting utilities
│   └── __init__.py
├── run_linear.py            # Experiments with linear regression (synthetic + California housing)
├── run_logistic.py          # Experiments with logistic regression (synthetic clf + breast cancer + digits)
├── run_digits_linear.py     # Digits 4 vs 9 classification (linear-as-classifier vs logistic)
├── results/
│   ├── linear/              # Output reports and plots for linear regression
│   ├── logistic/            # Output reports and plots for logistic regression
│   └── digits_linear_clf/   # Output reports for digits experiments
└── README.md
```

---

## Datasets

- **Synthetic Linear** – Simple y = w·x + noise regression dataset.
- **Synthetic Quadratic** – Extended to quadratic features for non-linear regression.
- **California Housing** – Real regression dataset from sklearn.
- **Synthetic Classification** – 2D synthetic dataset for binary classification visualization.
- **Breast Cancer** – Classification dataset from sklearn.
- **Digits (Full)** – Multiclass classification dataset (0–9).
- **Digits (4 vs 9)** – Subset reduced to binary classification.

---

## Methods Implemented

### Linear Regression
- Gradient descent (with L1/L2 regularization)
- Closed-form ridge regression (with safe epsilon for singular matrices)
- Bias term handling
- Standardization of features

### Logistic Regression
- Multiclass softmax regression via gradient descent
- L1 and L2 regularization
- Cross-validation hyperparameter tuning

### Digits Classification
- Binary classification (4 vs 9)
- Comparison of linear regression (with rounding wrapper) vs logistic regression

---

## Results

- All experiments log results into `results/` as `.txt` summaries and `.png` plots.
- Reports include:
  - Cross-validation sweeps for η (learning rate), λ₂ (ridge), λ₁ (lasso)
  - Final test MSEs (linear) or accuracies (logistic)
- Plots visualize:
  - CV sweep curves (Accuracy or MSE vs parameter)
  - Regression fits for synthetic linear/quadratic datasets
  - Classification regions for synthetic classification dataset

---

## How to Run

```bash
# Run linear regression experiments
python run_linear.py

# Run logistic regression experiments
python run_logistic.py

# Run digits (4 vs 9) experiments
python run_digits_linear.py
```

Outputs will be saved under `results/`.

---

## Notes

- Standardization parameters (μ, σ) are learned on the training set and reused for test preprocessing.
- Closed-form ridge uses a minimum epsilon (`1e-6`) to prevent singular matrices.
- Lasso closed-form is not implemented (requires coordinate descent); we only support lasso via gradient descent.

---

## Authors

Project completed as part of coursework. Contains implementations of linear models, logistic models, and dataset experiments with cross-validation and visualization.
