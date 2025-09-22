import numpy as np
from src.datasets import make_synth_clf, load_breast_cancer_data, load_digits_full, load_digits_4v9
from src.cv import cv_grid_search
from src.models_logistic import fit_gd_logistic, predict_labels_softmax, predict_scores_softmax
from src.utils import *
from src.plots import plot_curve
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

SEED = 123

n_train = 500
n_test = 1000

chosen_iters = 500

def run_grid(X:np.ndarray, y:np.ndarray, param_grid: dict, test_param: str, dataset_label: str) -> tuple[float,float,float,float]:
    """Helper function to run the cv_grid_search and return the best parameter value for the specified parameter

    Args:
        X (np.ndarray): input observations
        y (np.ndarray): expected outputs
        param_grid (dict): dictionary of parameters
        test_param (str): parameter cared about

    Returns:
        tuple[float,float]: best parameter, score of best, worst parameter, score of worst
    """
    if test_param not in param_grid.keys():
        raise ValueError(f"Invalid parameter specified: {test_param}")
    parameters, best, worst = cv_grid_search(fit_gd_logistic, predict_labels_softmax, X, y, param_grid, seed=SEED, task="clf")
    param_values_scores = [[params_score_pair["params"][test_param], params_score_pair["mean_metric"]] for params_score_pair in parameters]
    values = [pv[0] for pv in param_values_scores]
    scores = [pv[1] for pv in param_values_scores]
    plot_curve(values, scores, test_param, "Accuracy", f"Accuracy vs. {test_param}", f"results/logistic/logistic_results_{test_param}_{dataset_label}.png")
    return (best["params"][test_param], best["mean_metric"], worst["params"][test_param], worst["mean_metric"])


def main():
    # Sweep to find the best learning rate
    datasets = {"synthetic_clf": make_synth_clf(n_train, n_test, seed=SEED), "breast_cancer": load_breast_cancer_data(100,300,seed=SEED), "digits_full": load_digits_full(n_train,n_test,seed=SEED), "digits_4v9": load_digits_4v9(100,200,seed=SEED)}
    for dataset_name, dataset in datasets.items():
        X_train, y_train, X_test, y_test = dataset
        
        # Sweep to find the best learning rate
        param_grid_A = {"eta":[1e-3,3e-3,1e-2,3e-2,1e-1], "iters":[chosen_iters], "l1":[0.0], "l2":[0.0]}
        eta_star, accuracy_eta_star, worst_eta, accuracy_worst_eta = run_grid(X_train, y_train, param_grid_A, "eta", dataset_name)

        # Sweep to find best l2 regularization constant
        param_grid_B = {"eta":[eta_star], "iters":[chosen_iters], "l1":[0.0], "l2":[0.0,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2]}
        l2_star, accuracy_l2_star, worst_l2, accuracy_worst_l2 = run_grid(X_train, y_train, param_grid_B, "l2", dataset_name)
        
        # Now sweep to find best l1 regularization constant
        param_grid_C = {"eta":[eta_star], "iters":[chosen_iters], "l1":[0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2], "l2":[0]}
        l1_star, accuracy_l1_star, worst_l1, accuracy_worst_l1 = run_grid(X_train, y_train, param_grid_C, "l1", dataset_name)  
        
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
        report = f"""=== {dataset_name} ===
    CV-A (learning weight sweep, l2=0): best learning rate = {eta_star}, mean CV accuracy = {accuracy_eta_star}, worst eta = {worst_eta}, mean CV accuracy for worst eta = {accuracy_worst_eta}
    CV-B (l2 sweep @ eta={eta_star}): best l2 = {l2_star}, mean CV accuracy = {accuracy_l2_star}, worst l2 = {worst_l2}, mean CV accuracy for worst l2 = {accuracy_worst_l2}
    CV-C (l1 sweep @ eta={eta_star}): best l1 = {l1_star}, mean CV accuracy = {accuracy_l1_star}, worst l1 = {worst_l1}, mean CV accuracy for worst l1 = {accuracy_worst_l1}
    Accuracy without Regularized Learning: {accuracy_no_reg}
    Accuracy with Regularized Learning: {accuracy_reg}
    Accuracy with Lasso Learning: {accuracy_lasso}
        """
        
        os.makedirs("results/logistic", exist_ok=True)
        with open(f"results/logistic/{dataset_name}.txt", "w") as f:
            f.write(report)
            
        if dataset_name == "synthetic_clf":
            preds = {"no_reg": no_reg_model_weights, "reg": reg_model_weights, "lasso": lasso_model_weights}
            method_colors = {"no_reg": "C0", "reg": "C1", "lasso": "C2"}

            # Dense grid with a bit of padding
            x1_min, x1_max = X_test[:, 0].min(), X_test[:, 0].max()
            x2_min, x2_max = X_test[:, 1].min(), X_test[:, 1].max()
            pad1 = 0.05 * (x1_max - x1_min)
            pad2 = 0.05 * (x2_max - x2_min)
            xx, yy = np.meshgrid(
                np.linspace(x1_min - pad1, x1_max + pad1, 300),
                np.linspace(x2_min - pad2, x2_max + pad2, 300)
            )

            # Standardize with train mu/sigma and add bias
            grid = np.column_stack([xx.ravel(), yy.ravel()])
            grid_std = (grid - mu) / sigma
            grid_std = add_bias(grid_std)

            # Which index is the positive class?
            labels_sorted = np.unique(y_train)
            idx_pos = np.where(labels_sorted == 1)[0][0]
            idx_neg = 1 - idx_pos  # two-class case

            handles = []
            labels = []

            # First plot the points (slight transparency), then draw contours on top
            plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=18, alpha=0.7, zorder=1)

            for name, W in preds.items():
                scores = predict_scores_softmax(grid_std, W)     # shape (N, 2) of logits/scores
                diff = scores[:, idx_pos] - scores[:, idx_neg]   # positive minus negative
                Z = diff.reshape(xx.shape)

                cs = plt.contour(xx, yy, Z, levels=[0.0], colors=[method_colors[name]],
                                linewidths=2.5, zorder=3)
                handles.append(Line2D([0], [0], color=method_colors[name], lw=2.5))
                labels.append(name)

            plt.legend(handles=handles, labels=labels, title="Decision Boundary")
            plt.tight_layout()
            plt.savefig("results/logistic/synthetic_logistic_results.png")
            plt.close()
            
if __name__=="__main__":
    main()