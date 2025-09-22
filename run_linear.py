import numpy as np
from src.datasets import make_synth_reg_linear, make_synth_reg_quadratic, load_california
from src.cv import cv_grid_search
from src.models_linear import fit_gd_linear, fit_normal_eq_ridge, predict_linear
from src.utils import *
from src.plots import plot_curve
import os

SEED = 123

n_train = 500
n_test = 1000

chosen_iters = 500

def run_grid(X:np.ndarray, y:np.ndarray, param_grid: dict, test_param: str, label:str) -> tuple[float,float]:
    """Helper function to run the cv_grid_search and return the best parameter value for the specified parameter

    Args:
        X (np.ndarray): input observations
        y (np.ndarray): expected outputs
        param_grid (dict): dictionary of parameters
        test_param (str): parameter cared about
        label (str): data set label

    Returns:
        tuple[float,float]: the best value for parameter cared about paired with the resulting metric achieved by said value
    """
    if test_param not in param_grid.keys():
        raise ValueError(f"Invalid parameter specified: {test_param}")
    parameters, best, worst = cv_grid_search(fit_gd_linear, predict_linear, X, y, param_grid, seed=SEED)
    param_values_scores = [[params_score_pair["params"][test_param], params_score_pair["mean_metric"]] for params_score_pair in parameters]
    values = [pv[0] for pv in param_values_scores]
    scores = [pv[1] for pv in param_values_scores]
    plot_curve(values, scores, test_param, "MSE", f"MSE vs. {test_param}", f"results/linear/linear_results_{test_param}_{label}.png")
    return (best["params"][test_param], best["mean_metric"], worst["params"][test_param], worst["mean_metric"])

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
        eta_star, mse_eta_star, worst_eta, worst_eta_mse = run_grid(X_train, y_train, param_grid_A, "eta", label)

        # Sweep to find best l2 regularization constant
        param_grid_B = {"eta":[eta_star], "iters":[chosen_iters], "l1":[0.0], "l2":[0.0,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2]}
        l2_star, mse_l2_star, worst_l2, worst_l2_mse = run_grid(X_train, y_train, param_grid_B, "l2", label)
        
        # Now sweep to find best l1 regularization constant
        param_grid_C = {"eta":[eta_star], "iters":[chosen_iters], "l1":[0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2], "l2":[0]}
        l1_star, mse_l1_star, worst_l1, worst_l1_mse = run_grid(X_train, y_train, param_grid_C, "l1", label)  
        
        # The parameter grid sweep preprocesses the data, but for the remainder of our calls we need to do that ourself
        X_train, mu, sigma = standardize(X_train)
        X_train = add_bias(X_train)
        
        # Perform gradient descent with the best parameters we have discovered and NO regularization
        no_reg_model_weights = fit_gd_linear(X_train, y_train, eta_star, chosen_iters)
        
        # Now perform gradient descent with regularization
        reg_model_weights = fit_gd_linear(X_train, y_train, eta_star, chosen_iters, l2=l2_star)
        
        # Lasso gradient descent
        lasso_model_weights = fit_gd_linear(X_train, y_train, eta_star, chosen_iters, l1=l1_star)
        
        # Get your hands on the closed form ridge model (solved through linear algebra)
        l2_star_safe = max(l2_star, 1e-6) # To ensure non-singularity of matrix
        closed_form_ridge_weights = fit_normal_eq_ridge(X_train, y_train, l2_star_safe)
        
        # Now test all three of those models after preprocessing the test set
        X_test_std = (X_test - mu) / sigma
        X_test_std = add_bias(X_test_std)
        y_hat_no_reg = predict_linear(X_test_std, no_reg_model_weights)
        mse_no_reg = np.mean((y_hat_no_reg - y_test)**2)
        y_hat_reg = predict_linear(X_test_std, reg_model_weights)
        mse_reg = np.mean((y_hat_reg - y_test)**2)
        y_hat_lasso = predict_linear(X_test_std, lasso_model_weights)
        mse_lasso = np.mean((y_hat_lasso - y_test)**2)
        y_hat_closed = predict_linear(X_test_std, closed_form_ridge_weights)
        mse_closed = np.mean((y_hat_closed - y_test)**2)
        
        # Generate a report
        report = f"""=== {label} ===
    CV-A (learning weight sweep, l2=0): best learning rate = {eta_star}, mean CV MSE = {mse_eta_star}, worst eta = {worst_eta}, mean CV MSE worse eta = {worst_eta_mse}
    CV-B (l2 sweep @ eta={eta_star}): best l2 = {l2_star}, mean CV MSE = {mse_l2_star}, worst l2 = {worst_l2}, mean CV MSE worse l2 = {worst_l2_mse}
    CV-C (l1 sweep @ eta={eta_star}): best l1 = {l1_star}, mean CV MSE = {mse_l1_star}, worst l1 = {worst_l1}, mean CV MSE worse l1 = {worst_l1_mse}
    Test MSEs:
    GD (no reg):    {mse_no_reg}
    GD (ridge, l2_star):  {mse_reg}
    GD (lasso, l1_star): {mse_lasso}
    Closed-form ridge (l2_star): {mse_closed}
        """
        
        os.makedirs("results/linear", exist_ok=True)
        with open(f"results/linear/{label}.txt", "w") as f:
            f.write(report)
            
        # We'll also create plots for the synthetic linear and linearized synthetic quadratic plots
        if label == "synthetic_linear" or label == "linearized_synth_quadratic":
            x = X_test.squeeze()
            if label == "linearized_synth_quadratic":
                x = X_test[:, 0] # Only care about actual features for plotting; not squared features
            order = np.argsort(x)
            x = x[order]
            preds = {"no_reg":y_hat_no_reg[order], "reg":y_hat_reg[order], "lasso":y_hat_lasso[order], "closed":y_hat_closed[order]}
            for title, yhat in preds.items():
                plot_curve(x, yhat, "X", "yhat", f"{title} on {label}", f"results/linear/{title}_linear_regression_{label}.png")
            
if __name__=="__main__":
    main()