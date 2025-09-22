import numpy as np
from src.datasets import load_digits_4v9
from src.cv import cv_grid_search
from src.models_linear import fit_gd_linear, fit_normal_eq_ridge
from src.models_logistic import fit_gd_logistic, predict_labels_softmax
from src.utils import *

SEED = 123

n_train = 100
n_test = 200

chosen_iters = 500

def run_grid(X:np.ndarray, y:np.ndarray, param_grid: dict, test_param: str) -> tuple[float,float]:
    """Helper function to run the cv_grid_search and return the best parameter value for the specified parameter

    Args:
        X (np.ndarray): input observations
        y (np.ndarray): expected outputs
        param_grid (dict): dictionary of parameters
        test_param (str): parameter cared about

    Returns:
        tuple[float,float]: the best value for parameter cared about paired with the resulting metric achieved by said value
    """
    if test_param not in param_grid.keys():
        raise ValueError(f"Invalid parameter specified: {test_param}")
    _, best, _ = cv_grid_search(fit_gd_linear, linear_digits4v9_clf_wrapper, X, y, param_grid, seed=SEED, task="clf")
    return (best["params"][test_param], best["mean_metric"])

def run_grid_logistic(X:np.ndarray, y:np.ndarray, param_grid: dict, test_param: str) -> tuple[float,float]:
    """Helper function to run the cv_grid_search and return the best parameter value for the specified parameter

    Args:
        X (np.ndarray): input observations
        y (np.ndarray): expected outputs
        param_grid (dict): dictionary of parameters
        test_param (str): parameter cared about

    Returns:
        tuple[float,float]: the best value for parameter cared about paired with the resulting metric achieved by said value
    """
    if test_param not in param_grid.keys():
        raise ValueError(f"Invalid parameter specified: {test_param}")
    _, best, _ = cv_grid_search(fit_gd_logistic, predict_labels_softmax, X, y, param_grid, seed=SEED, task="clf")
    return (best["params"][test_param], best["mean_metric"])

def linear_digits4v9_clf_wrapper(X: np.ndarray, W: np.ndarray, task=None) -> np.ndarray:
    """Helper function to take the linear predictions of a trained linear regression model and round to use it as binary classification

    Args:
        X (np.ndarray): Input data
        W (np.ndarray): Linearly trained weights
        task: Unused parameter to be ignored but making the pass into cv_grid_search compatible

    Returns:
        np.ndarray: Resulting rounded classification predictions
    """
    y_hat = np.matmul(X, W)
    y_hat = np.clip(y_hat, 0, 1)
    return np.rint(y_hat).astype(int).reshape(-1)

def main():
    # Sweep to find the best learning rate
    dataset = load_digits_4v9(n_train,n_test,seed=SEED)
    X_train, y_train, X_test, y_test = dataset
    
    # Sweep to find the best learning rate
    param_grid_A = {"eta":[1e-3,3e-3,1e-2,3e-2,1e-1], "iters":[chosen_iters], "l1":[0.0], "l2":[0.0]}
    eta_star_linear, accuracy_eta_star_linear = run_grid(X_train, y_train, param_grid_A, "eta")

    # Sweep to find best l2 regularization constant
    param_grid_B = {"eta":[eta_star_linear], "iters":[chosen_iters], "l1":[0.0], "l2":[0.0,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2]}
    l2_star_linear, accuracy_l2_star_linear = run_grid(X_train, y_train, param_grid_B, "l2")
    
    # Now sweep to find best l1 regularization constant
    param_grid_C = {"eta":[eta_star_linear], "iters":[chosen_iters], "l1":[0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2], "l2":[0]}
    l1_star_linear, accuracy_l1_star_linear = run_grid(X_train, y_train, param_grid_C, "l1")
    
    # The parameter grid sweep preprocesses the data, but for the remainder of our calls we need to do that ourself
    X_train_std, mu, sigma = standardize(X_train)
    X_train_std = add_bias(X_train_std)
    
    # Perform gradient descent with the best parameters we have discovered and NO regularization
    no_reg_model_weights = fit_gd_linear(X_train_std, y_train, eta_star_linear, chosen_iters)
    
    # Now perform gradient descent with regularization
    reg_model_weights = fit_gd_linear(X_train_std, y_train, eta_star_linear, chosen_iters, l2=l2_star_linear)
    
    # Now with lasso
    lasso_model_weights = fit_gd_linear(X_train_std, y_train, eta_star_linear, chosen_iters, l1=l1_star_linear)
    
    # Get your hands on the closed form ridge model for l2 (solved through linear algebra)
    l2_star_ridge = max(l2_star_linear, 1e-6) # prevent singular matrix
    closed_form_ridge_weights_reg = fit_normal_eq_ridge(X_train_std, y_train, l2_star_ridge)
    
   
    # Now test all three of those models after preprocessing the test set
    X_test_std = (X_test - mu) / sigma
    X_test_std = add_bias(X_test_std)
    predictions_no_reg = linear_digits4v9_clf_wrapper(X_test_std, no_reg_model_weights)
    accuracy_no_reg = np.sum(predictions_no_reg == y_test) / len(y_test)
    predictions_reg = linear_digits4v9_clf_wrapper(X_test_std, reg_model_weights)
    accuracy_reg = np.sum(predictions_reg == y_test) / len(y_test)
    predictions_lasso = linear_digits4v9_clf_wrapper(X_test_std, lasso_model_weights)
    accuracy_lasso = np.sum(predictions_lasso == y_test) / len(y_test)
    predictions_closed_reg = linear_digits4v9_clf_wrapper(X_test_std, closed_form_ridge_weights_reg)
    accuracy_closed_reg = np.sum(predictions_closed_reg == y_test) / len(y_test)
    
    label = "digits4v9_linear_binary_classification_wrapper"
    
    # Generate a report
    report = f"""=== {label} ===
CV-A (learning weight sweep, l2=0): best learning rate = {eta_star_linear}, mean CV Accuracy = {accuracy_eta_star_linear}
CV-B (l2 sweep @ eta={eta_star_linear}): best l2 = {l2_star_linear}, mean CV Accuracy = {accuracy_l2_star_linear}
CV-C (l1 sweep @ eta={eta_star_linear}): best l1 = {l1_star_linear}, mean CV Accuracy = {accuracy_l1_star_linear}
Test Accuracies:
GD (no reg):    {accuracy_no_reg}
GD (ridge, l2_star):  {accuracy_reg}
GD (lasso, l1_star): {accuracy_lasso}
Closed-form ridge (l2_star): {accuracy_closed_reg}
        """
        
    import os
    os.makedirs("results/digits_linear_clf", exist_ok=True)
    with open(f"results/digits_linear_clf/{label}.txt", "w") as f:
        f.write(report)
        
    # Now we simply perform logistic regression on the digits via binary classification normally and see how it goes
    # Sweep to find the best learning rate
    param_grid_A = {"eta":[1e-3,3e-3,1e-2,3e-2,1e-1], "iters":[chosen_iters], "l1":[0.0], "l2":[0.0]}
    eta_star_logistic, accuracy_eta_star_logistic = run_grid_logistic(X_train, y_train, param_grid_A, "eta")

    # Sweep to find best l2 regularization constant
    param_grid_B = {"eta":[eta_star_logistic], "iters":[chosen_iters], "l1":[0.0], "l2":[0.0,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2]}
    l2_star_logistic, accuracy_l2_star_logistic = run_grid_logistic(X_train, y_train, param_grid_B, "l2")
    
    # Sweep to find best l1 regularization constant
    param_grid_C = {"eta":[eta_star_logistic], "iters":[chosen_iters], "l1":[0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2], "l2":[0]}
    l1_star_logistic, accuracy_l1_star_logistic = run_grid_logistic(X_train, y_train, param_grid_C, "l1")
    
    # Perform gradient descent with the best parameters we have discovered and NO regularization
    K = len(np.unique(y_train))
    no_reg_model_weights = fit_gd_logistic(X_train_std, y_train, K, eta_star_logistic, chosen_iters)
    
    # Test the non-regularization training logistic model after preprocessing the test set
    y_hat_no_reg = predict_labels_softmax(X_test_std, no_reg_model_weights)
    accuracy_no_reg = sum(y_hat_no_reg == y_test) / len(y_test)
    
    # Now for regularization training
    reg_model_weights = fit_gd_logistic(X_train_std, y_train, K, eta_star_logistic, chosen_iters, l2=l2_star_logistic)
    y_hat_reg = predict_labels_softmax(X_test_std, reg_model_weights)
    accuracy_reg = sum(y_hat_reg == y_test) / len(y_test)
    
    # Now for lasso training
    lasso_model_weights = fit_gd_logistic(X_train_std, y_train, K, eta_star_logistic, chosen_iters, l1=l1_star_logistic)
    y_hat_lasso = predict_labels_softmax(X_test_std, lasso_model_weights)
    accuracy_lasso = sum(y_hat_lasso == y_test) / len(y_test)
    
    label = "digits4v9_normal_binary_clf"
    
    # Generate a report
    report = f"""=== {label} ===
CV-A (learning weight sweep, l2=0): best learning rate = {eta_star_logistic}, mean CV accuracy = {accuracy_eta_star_logistic}
CV-B (l2 sweep @ eta={eta_star_logistic}): best l2 = {l2_star_logistic}, mean CV accuracy = {accuracy_l2_star_logistic}
CV-C (l1 sweep @ eta={eta_star_logistic}): best l1 = {l1_star_logistic}, mean CV accuracy = {accuracy_l1_star_logistic}
Accuracy without Regularized Learning: {accuracy_no_reg}
Accuracy with Regularized Learning: {accuracy_reg}
Accuracy with Lasso Learning: {accuracy_lasso}
    """
    
    with open(f"results/digits_linear_clf/{label}.txt", "w") as f:
        f.write(report)
            
if __name__=="__main__":
    main()