import numpy as np

def add_bias(X: np.ndarray) -> np.ndarray: # We'll just prepend a column of 1's
    if len(X.shape) == 1:
        X = np.reshape(X, (X.shape[0],1))
    elif len(X.shape) != 2:
        raise ValueError(f"X has unexpected dimensions of {X.shape}")
  
    n = X.shape[0]
    ones = np.ones(shape=(n,1), dtype=float)
    X = np.hstack([ones, X])
    return X

def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]: # For consistency across folds
    if len(X.shape) == 1:
        X = X.reshape(shape=(X.shape[0],1))
    n = X.shape[0]
    mu = np.mean(X, axis=0) # Average of the features over all observations
    sigma = np.std(X, axis=0) # Standard deviation of the features over all observations
    sigma[sigma == 0.0] = 1.0 # Slick divide by zero correction that the internet gave me
    X_std = (X - mu) / sigma
    assert X_std.shape == (n,)
    return (X_std, mu, sigma)

def train_test_sample(X: np.ndarray, y: np.ndarray,  n_train: int, n_test: int, seed: int): # Sample without replacement
    pass

def soft_threshold(w: np.ndarray, alpha: np.float64): # For L1 updates in a proximal step
    pass