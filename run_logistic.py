import numpy as np
from src.datasets import make_synth_clf, load_breast_cancer_data, load_digits_full, load_digits_4v9
from src.cv import cv_grid_search
from src.models_logistic import fit_gd_logistic, predict_labels_softmax
from src.utils import *
from src.plots import plot_curve
import matplotlib.pyplot as plt
import os

SEED = 123

n_train = 500
n_test = 1000

chosen_iters = 500

def run_grid(X:np.ndarray, y:np.ndarray, param_grid: dict, test_param: str, dataset_label: str) -> tuple[float,float]:
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
    parameters, best, _ = cv_grid_search(fit_gd_logistic, predict_labels_softmax, X, y, param_grid, seed=SEED, task="clf")
    param_values_scores = [[params_score_pair["params"][test_param], params_score_pair["mean_metric"]] for params_score_pair in parameters]
    values = [pv[0] for pv in param_values_scores]
    scores = [pv[1] for pv in param_values_scores]
    plot_curve(values, scores, test_param, "Accuracy", f"Accuracy vs. {test_param}", f"results/logistic/logistic_results_{test_param}_{dataset_label}.png")
    return (best["params"][test_param], best["mean_metric"])


def main():
    # Sweep to find the best learning rate
    datasets = {"synthetic_clf": make_synth_clf(n_train, n_test, seed=SEED), "breast_cancer": load_breast_cancer_data(100,300,seed=SEED), "digits_full": load_digits_full(n_train,n_test,seed=SEED), "digits_4v9": load_digits_4v9(100,200,seed=SEED)}
    for label, dataset in datasets.items():
        X_train, y_train, X_test, y_test = dataset
        
        # Sweep to find the best learning rate
        param_grid_A = {"eta":[1e-3,3e-3,1e-2,3e-2,1e-1], "iters":[chosen_iters], "l1":[0.0], "l2":[0.0]}
        eta_star, accuracy_eta_star = run_grid(X_train, y_train, param_grid_A, "eta", label)

        # Sweep to find best l2 regularization constant
        param_grid_B = {"eta":[eta_star], "iters":[chosen_iters], "l1":[0.0], "l2":[0.0,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2]}
        l2_star, accuracy_l2_star = run_grid(X_train, y_train, param_grid_B, "l2", label)
        
        # Now sweep to find best l1 regularization constant
        param_grid_C = {"eta":[eta_star], "iters":[chosen_iters], "l1":[0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2], "l2":[0]}
        l1_star, accuracy_l1_star = run_grid(X_train, y_train, param_grid_C, "l1", label)  
        
        # The parameter grid sweep preprocesses the data, but for the remainder of our calls we need to do that ourself
        X_train, mu, sigma = standardize(X_train)
        X_train = add_bias(X_train)
        
        # Perform gradient descent with the best parameters we have discovered and NO regularization
        K = len(np.unique(y_train))
        no_reg_model_weights = fit_gd_logistic(X_train, y_train, K, eta_star, chosen_iters)
        
        # Test the non-regularization training logistic model after preprocessing the test set
        X_test_std = (X_test - mu) / sigma
        X_test_std = add_bias(X_test_std)
        y_hat_no_reg = predict_labels_softmax(X_test_std, no_reg_model_weights)
        accuracy_no_reg = sum(y_hat_no_reg == y_test) / len(y_test)
        
        # Now for regularization training
        reg_model_weights = fit_gd_logistic(X_train, y_train, K, eta_star, chosen_iters, l2=l2_star)
        y_hat_reg = predict_labels_softmax(X_test_std, reg_model_weights)
        accuracy_reg = sum(y_hat_reg == y_test) / len(y_test)
        
        # Lasso training
        lasso_model_weights = fit_gd_logistic(X_train, y_train, K, eta_star, chosen_iters, l1=l1_star)
        y_hat_lasso = predict_labels_softmax(X_test_std, lasso_model_weights)
        accuracy_lasso = sum(y_hat_lasso == y_test) / len(y_test)
        
        # Generate a report
        report = f"""=== {label} ===
    CV-A (learning weight sweep, l2=0): best learning rate = {eta_star}, mean CV accuracy = {accuracy_eta_star}
    CV-B (l2 sweep @ eta={eta_star}): best l2 = {l2_star}, mean CV accuracy = {accuracy_l2_star}
    CV-C (l1 sweep @ eta={eta_star}): best l1 = {l1_star}, mean CV accuracy = {accuracy_l1_star}
    Accuracy without Regularized Learning: {accuracy_no_reg}
    Accuracy with Regularized Learning: {accuracy_reg}
    Accuracy with Lasso Learning: {accuracy_lasso}
        """
        
        os.makedirs("results/logistic", exist_ok=True)
        with open(f"results/logistic/{label}.txt", "w") as f:
            f.write(report)
            
        # We'll also create plots for the 2D synthetic classifiable data
        if label == "synthetic_clf":
            preds = {"no_reg":y_hat_no_reg, "reg":y_hat_reg, "lasso":y_hat_lasso}
            for title, yhat in preds.items():
                plt.scatter(X_test[:,0], X_test[:,1], c=yhat)
                plt.title(f"{title} on {label}")
                plt.savefig(f"results/logistic/{title}_logistic_regression_{label}.png")
                plt.close()
            
if __name__=="__main__":
    main()