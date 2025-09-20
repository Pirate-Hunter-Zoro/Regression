import numpy as np
from src.datasets import make_synth_reg_linear, make_synth_reg_quadratic, load_california
from src.cv import cv_grid_search
from src.models_linear import fit_gd_linear, fit_normal_eq_ridge, predict_linear
from src.utils import *

SEED = 123

n_train = 500
n_test = 1000

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
    _, best, _ = cv_grid_search(fit_gd_linear, predict_linear, X, y, param_grid, seed=SEED)
    return (best["params"][test_param], best["mean_metric"])

def linearize_synth_quad(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linearize synthetic quadratic data by appending the features squared to the list of features (doubling the length of the features) 

    Args:
        X_train (np.ndarray): synthetic quadratic training data
        X_test (np.ndarray): synthetic quadratic testing data

    Returns:
        tuple[np.ndarray, np.ndarray]: resulting quadratic training and testing data that has been linearized
    """
    if len(X_train.shape) == 1:
        X_train = X_train.reshape((len(X_train),1))
    X_train_transformed = np.hstack([X_train, X_train ** 2])
    if not X_train_transformed.shape == (len(X_train),2*X_train.shape[1]):
        raise ValueError(f"Created unexpected linearized data dimensions: {X_train_transformed.shape}")
    
    # Transform X_test
    if len(X_test.shape) == 1:
        X_test = X_test.reshape((len(X_test),1))
    X_test_transformed = np.hstack([X_test, X_test ** 2])
    
    return X_train_transformed, X_test_transformed

def main():
    # Sweep to find the best learning rate
    datasets = {"synthetic_linear": make_synth_reg_linear(n_train, n_test, seed=SEED), "synthetic_quadratic": make_synth_reg_quadratic(n_train,n_test,seed=SEED), "california_housing": load_california(n_train,n_test,seed=SEED)}
    X_train, y_train, X_test, y_test = datasets["synthetic_quadratic"]
    X_train, X_test = linearize_synth_quad(X_train, X_test)
    datasets["linearized_synth_quadratic"]=(X_train, y_train, X_test, y_test)
    for label, dataset in datasets.items():
        X_train, y_train, X_test, y_test = dataset
        
        # Sweep to find the best learning rate
        param_grid_A = {"eta":[1e-3,3e-3,1e-2,3e-2,1e-1], "iters":[chosen_iters], "l1":[0.0], "l2":[0.0]}
        eta_star, mse_eta_star = run_grid(X_train, y_train, param_grid_A, "eta")

        # Sweep to find best l2 regularization constant
        param_grid_B = {"eta":[eta_star], "iters":[chosen_iters], "l1":[0.0], "l2":[0.0,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2]}
        l2_star, mse_l2_star = run_grid(X_train, y_train, param_grid_B, "l2")
        
        # The parameter grid sweep preprocesses the data, but for the remainder of our calls we need to do that ourself
        X_train, mu, sigma = standardize(X_train)
        X_train = add_bias(X_train)
        
        # Perform gradient descent with the best parameters we have discovered and NO regularization
        no_reg_model_weights = fit_gd_linear(X_train, y_train, eta_star, chosen_iters)
        
        # Now perform gradient descent with regularization
        reg_model_weights = fit_gd_linear(X_train, y_train, eta_star, chosen_iters, l2=l2_star)
        
        # Get your hands on the closed form ridge model (solved through linear algebra)
        closed_form_ridge_weights = fit_normal_eq_ridge(X_train, y_train, l2_star)
        
        # Now test all three of those models after preprocessing the test set
        X_test = (X_test - mu) / sigma
        X_test = add_bias(X_test)
        y_hat_no_reg = predict_linear(X_test, no_reg_model_weights)
        mse_no_reg = np.mean((y_hat_no_reg - y_test)**2)
        y_hat_reg = predict_linear(X_test, reg_model_weights)
        mse_reg = np.mean((y_hat_reg - y_test)**2)
        y_hat_closed = predict_linear(X_test, closed_form_ridge_weights)
        mse_closed = np.mean((y_hat_closed - y_test)**2)
        
        # Generate a report
        report = f"""=== {label} ===
    CV-A (learning weight sweep, l2=0): best learning rate = {eta_star}, mean CV MSE = {mse_eta_star}
    CV-B (l2 sweep @ eta={eta_star}): best l2 = {l2_star}, mean CV MSE = {mse_l2_star}
    Test MSEs:
    GD (no reg):    {mse_no_reg}
    GD (ridge, l2_star):  {mse_reg}
    Closed-form ridge (l2_star): {mse_closed}
        """
        
        import os
        os.makedirs("results/linear", exist_ok=True)
        with open(f"results/linear/{label}.txt", "w") as f:
            f.write(report)
            
if __name__=="__main__":
    main()