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

def cv_grid_search(model_fn, predict_fn, X:np.ndarray, y:np.ndarray, k:int, param_grid, task):
    pass