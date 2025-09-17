import numpy as np
from src.utils import *

def kfold_indices(n: int, k:int=10, seed:int=42) -> list[tuple[np.ndarray,np.ndarray]]:
    """Create a (seeded) random partioning of the observation indices in k folds, where during training each fold gets a chance to be the test set and all other folds form the training set

    Args:
        n (int): total number of observations
        k (int, optional): Number of partitions. Defaults to 10.
        seed (int, optional): random seed to make split deterministic. Defaults to 42.

    Raises:
        ValueError: if invalid population/partition count values are specified

    Returns:
        list[tuple[np.ndarray,np.ndarray]]: For a given fold, specifies which indices are the train set and which indices are the test set
    """
    # We need random partitions of size k
    shuffled_indices = np.random.RandomState(seed).permutation(n).tolist()
    if n < k:
        raise ValueError(f"Cannot create {k} partitions out of sample size {n}")
    partition_size = n // k
    partitions = []
    # The first n % k partitions will get an extra
    plus_one = n % k
    left = 0
    right = partition_size # Right index is exclusive
    while right <= n:
        if plus_one:
            # We are still in the first n % k partitions that need an extra observation
            plus_one -= 1
            right += 1
        train_split = []
        test_split = shuffled_indices[left:right]
        if left > 0:
            train_split += shuffled_indices[:left]
        if right < n:
            train_split += shuffled_indices[right:]
        train_split = np.array(train_split)
        test_split = np.array(test_split)
        partitions.append((train_split, test_split))
        left = right
        right = left + partition_size
    
    return partitions

def cv_grid_search(model_fn, predict_fn, X:np.ndarray, y:np.ndarray, param_grid, *, k:int=10, task:str="reg") -> tuple[list[dict], dict, dict]:
    """For each hyperparameter setting in the grid, run k-fold cross-validation and record the mean score.
    The hyperparameter setting with the best average score will be picked.

    Args:
        model_fn (function): model_fit_fn(X_tr_biased, y_tr, **params) -> model
        predict_fn (function): predict_fun(X_val_biased, model, task) -> prediction (for regression, y_pred has shape (m,); for classification, still same shape but integer labels)
        X (np.ndarray): input data
        y (np.ndarray): expected outputs
        param_grid (dict): dict where each key is a hyperparameter name and each value is a list of candidate values to test
        k (int, optional): k-fold specification. Defaults to 10.
        task (str, optional): dictates classification or regression. Defaults to "reg".

    Returns:
        tuple[list[dict], dict, dict]: list[{"params": <dict>, "mean_metric": <float>}], best dict, worst dict
    """
    if not len(X.shape) == 2:
        raise ValueError(f"Input X should be 2D but had shape {X.shape}")
    n = X.shape[0]
    if not y.shape == (n,):
        raise ValueError(f"Expected output y should have shape {(n,)} but instead had shape {y.shape}")
    if k < 2:
        raise ValueError(f"Expected at least 2 folds but recieved k={k}")
    if n < k:
        raise ValueError(f"Cannot perform k-fold cross validation when k>n, recieved k={k} and n={n}")
    if task != "reg" and task != "clf":
        raise ValueError(f"Recieved invalid task argument: {task}")
    
    parameter_keys = set(["eta", "iters", "l1", "l2", "standardize"])
    for v in parameter_keys:
        if v not in param_grid.keys():
            raise ValueError(f"Missing parameter value: {v}")
    for k, v in param_grid:
        # Ensure iterable
        if not isinstance(v, iter):
            raise ValueError(f"Invalid parameter list specified: {v}")
        if k not in parameter_keys:
            raise ValueError(f"Invalid parameter key specified: {k}")
    
    results = []
    k_folds = kfold_indices(n, k)
    for eta in param_grid["eta"]:
        for iters in param_grid["iters"]:
            for l2 in param_grid["l1"]:
                for l1 in param_grid["l2"]:
                    for standardize in param_grid["standardize"]:
                        params = {"eta":eta, "iters":iters, "l1":l1, "l2":l2, "standardize":standardize}
                        fold_metrics = []
                        for train_test_split in k_folds:
                            X_tr = X[train_test_split[0]]
                            y_tr = y[train_test_split[0]]
                            X_val = X[train_test_split[1]]
                            y_val = y[train_test_split[1]]
                            
                            if standardize:
                                X_tr = standardize(X_tr)
                                X_val = standardize(X_val)
                            
                            X_tr = add_bias(X_tr)
                            X_val = add_bias(X_val)
                            
                            model = model_fn(X_tr, y_tr, eta, iters, l1, l2)
                            
                            pred = predict_fn(X_val, model, task)
                            if pred.shape != y.shape:
                                raise ValueError(f"Received invalid prediction output of shape {pred.shape} when expecting {y.shape}")
                            
                            # See how the model did
                            if task == "reg":
                                # Squared error
                                mse = np.mean((y_val-pred)**2).item()
                                fold_metrics.append(mse)
                            else:
                                # Accuracy count
                                accuracy = np.sum((y_val == pred))/n
                                fold_metrics.append(accuracy)
                        
                        # All folds for this combo of parameters are done
                        mean_metric = sum(fold_metrics)/len(fold_metrics)
                        results.append({"params":params, "mean_metric":mean_metric})
    
    mean_metrics = np.array([result["mean_metric"] for result in results])
    if task == "reg":
        best_idx = np.argmin(mean_metrics)
        worst_idx = np.argmin(mean_metrics)
    else:
        best_idx = np.argmax(mean_metrics)
        worst_idx = np.argmin(mean_metrics)
    best = results[best_idx]["params"]
    worst = results[worst_idx]["params"]
    return (results, best, worst)