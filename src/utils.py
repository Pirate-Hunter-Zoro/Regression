import numpy as np

def add_bias(X: np.ndarray) -> np.ndarray: # We'll just prepend a column of 1's
    """Prepend a column of 1's to X so that when the weight vector is learned, the first term simply becomes the bias

    Args:
        X (np.ndarray): Input data

    Returns:
        np.ndarray: Input data with bias vector prepended
    """
    if len(X.shape) == 1:
        X = np.reshape(X, (X.shape[0],1))
    elif len(X.shape) != 2:
        raise ValueError(f"X has unexpected dimensions of {X.shape}")
  
    n = X.shape[0]
    ones = np.ones(shape=(n,1), dtype=float)
    X = np.hstack([ones, X])
    return X

def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]: # For consistency across folds
    """Standardize each numerical feature of the observations over all observations

    Args:
        X (np.ndarray): Input observations

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Standardized observations, respective mean of each feature, respective standard deviation of each feature
    """
    if len(X.shape) == 1:
        X = np.reshape(X, shape=(X.shape[0],1))
    n = X.shape[0]
    k = X.shape[1]
    mu = np.mean(X, axis=0) # Average of the features over all observations
    sigma = np.std(X, axis=0) # Standard deviation of the features over all observations
    sigma[sigma == 0.0] = 1.0 # Slick divide by zero correction that the internet gave me
    X_std = (X - mu) / sigma
    if not X_std.shape == (n,k):
        raise ValueError(f"Expected standardized array dimensions to be {(n,k)} but observed shape {X_std.shape}")
    return (X_std, mu, sigma)

def train_test_sample(X: np.ndarray, y: np.ndarray,  n_train: int, n_test: int, seed: int=42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: # Sample without replacement
    """Helper function to sample training and testing indices out of input given observations and traing and test sizes

    Args:
        X (np.ndarray): Input observations
        y (np.ndarray): Output values
        n_train (int): Number of training instances desired
        n_test (int): Number of testing instances desired
        seed (int, optional): Random seed for deterministic results. Defaults to 42.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, y_train, X_test, y_test
    """
    if len(X.shape) == 1:
        X = np.reshape(X, shape=(X.shape[0],1))
    n = X.shape[0]
    if not y.shape == (n,):
        raise ValueError(f"Input y shape does not match the number of input observations and instead has shape {y.shape}")
    if not (n_train > 0 and (isinstance(n_train, int) or isinstance(n_train, np.int))):
        raise ValueError("Need positive number of training examples")
    if not (n_test > 0 and (isinstance(n_train, int) or isinstance(n_train, np.int))): 
        raise ValueError("Need positive number of testing examples")
    if not (n_train + n_test <= n):
        raise ValueError("Number of training and testing samples must not exceed n")
    rng = np.random.RandomState(seed=seed)
    indices_perm = rng.permutation(n)
    train_indices = indices_perm[:n_train]
    test_indices = indices_perm[n_train:n_train+n_test]
    train_idx_set = set(train_indices)
    test_idx_set = set(test_indices)
    for i in train_idx_set:
        if i in test_idx_set:
            raise ValueError("Training and testing sset are not mutually exclusive")
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    return (X_train, y_train, X_test, y_test)

def soft_threshold(w: np.ndarray, alpha: np.float64) -> np.ndarray: # For L1 updates in a proximal step
    """Scale all elements in w to have an absolute value of max(0, abs(element)-alpha)

    Args:
        w (np.ndarray): elements to scale
        alpha (np.float64): scale value

    Returns:
        np.ndarray: scaled elements
    """
    abs_vals = np.abs(w)
    factors = (abs_vals - alpha)
    factors[factors < 0] = 0
    w = np.sign(w) * factors
    return w