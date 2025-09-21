# Policy Discovery – Regression & Classification

This project implements regression and classification algorithms from scratch, with cross-validation, hyperparameter sweeps, and dataset evaluations. It is part of the CS885 assignment framework.

## Contents

- **src/**: Core source code  
  - `datasets.py`: Synthetic and real dataset loaders (California Housing, Breast Cancer, Digits, etc.)  
  - `cv.py`: Cross-validation grid search (`cv_grid_search`)  
  - `models_linear.py`: Linear regression (gradient descent with L1/L2, normal equation with ridge)  
  - `models_logistic.py`: Logistic regression (gradient descent with L1/L2)  
  - `plots.py`: Simple plotting utilities (`plot_curve`)  
  - `utils.py`: Bias column, standardization, helpers  

- **run_linear.py**: Runs regression experiments (synthetic linear, quadratic, California housing)  
- **run_logistic.py**: Runs classification experiments (synthetic classification, breast cancer, digits full, digits 4v9)  
- **run_digits_linear.py**: Special comparison between linear regression (with wrapper) vs. logistic regression on Digits 4v9  
- **results/**:  
  - `linear/`: Text reports + plots for regression experiments  
  - `logistic/`: Text reports + plots for classification experiments  
  - `digits_linear_clf/`: Text reports + plots for 4v9 linear vs. logistic comparison  

## Requirements

- Python 3.10+  
- NumPy  
- Matplotlib  
- scikit-learn (for datasets)  

Create a virtual environment and install requirements:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

*(Add a `requirements.txt` if needed, listing `numpy`, `matplotlib`, `scikit-learn`.)*

## Usage

### Run Regression Experiments

```bash
python run_linear.py
```

Outputs:

- Text reports in `results/linear/`  
- Plots:  
  - MSE vs. learning rate (`linear_results_eta.png`)  
  - MSE vs. regularization (`linear_results_l2.png`)  
  - Predictor fits for synthetic datasets  

### Run Classification Experiments

```bash
python run_logistic.py
```

Outputs:

- Text reports in `results/logistic/`  
- Plots:  
  - Accuracy vs. learning rate (`logistic_results_eta_*.png`)  
  - Accuracy vs. regularization (`logistic_results_l2_*.png`)  
  - Synthetic classification decision boundaries  

### Run Digits Linear vs Logistic Comparison

```bash
python run_digits_linear.py
```

Outputs:

- Text reports in `results/digits_linear_clf/`  
- Plots comparing linear regression wrapper vs logistic regression on 4v9  

## Project Notes

- All models are trained from scratch using gradient descent or closed-form solutions.  
- Cross-validation (`cv_grid_search`) is used to select hyperparameters (`eta`, `l2`).  
- Bias terms must be added explicitly (`add_bias` from `utils.py`).  
- For quadratic regression, input features are linearized ([X, X²]).  

## Results Overview

- **Regression**:  
  - Linear fits show expected performance (quadratic improves after linearization).  
  - Ridge regularization reduces variance slightly.  
- **Classification**:  
  - Logistic regression achieves high accuracy across datasets.  
  - Digits 4v9: logistic regression outperforms linear regression with wrapper.  

---

## Authors

Implemented for CS885 Assignment – Winter 2022.  
