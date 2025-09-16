import numpy as np

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
        param_grid (dict): dict wehre each key is a hyperparameter name and each value is a list of candidate values to test
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
    
    for _, v in param_grid:
        # Ensure iterable
        if not isinstance(v, iter):
            raise ValueError(f"Invalid parameter list specified: {v}")