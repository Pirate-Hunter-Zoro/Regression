import numpy as np

def make_synth_reg_linear(n_train: int=100, n_test: int=20, noise: float=0.1, seed: int=42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
    """Generate synthetic linear regression data with the specified number of training and testing cases and noise

    Args:
        n_train (int, optional): Number of training instances. Defaults to 100.
        n_test (int, optional): Number of testing instances. Defaults to 20.
        noise (float, optional): Noise to keep the y_values from being boring and perfectly linear. Defaults to 0.1.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, y_train, X_test, y_test
    """
    rng = np.random.RandomState(seed=seed)
    n = n_train + n_test
    X_all = rng.rand(n) # all between 0 and 1
    a = 3
    b = 5
    no_noise = a*X_all + b
    y_all = no_noise + noise*rng.randn(n)
    shuffled_indices = rng.permutation(n)
    train_indices = shuffled_indices[:n_train]
    test_indices = shuffled_indices[n_train:]
    
    # Now break up the two inputs and outputs
    X_train = X_all[train_indices]
    y_train = y_all[train_indices]
    X_test = X_all[test_indices]
    y_test = y_all[test_indices]
    return (X_train, y_train, X_test, y_test)

def make_synth_reg_quadratic(n_train: int=100, n_test: int=20, noise: float=0.1, seed: int=42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
    """Generate synthetic quadratic regression data with the specified number of training and testing cases and noise

    Args:
        n_train (int, optional): Number of training instances. Defaults to 100.
        n_test (int, optional): Number of testing instances. Defaults to 20.
        noise (float, optional): Noise to keep the y_values from being boring and perfectly linear. Defaults to 0.1.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, y_train, X_test, y_test
    """
    rng = np.random.RandomState(seed=seed)
    n = n_train + n_test
    X_all = rng.rand(n) # all between 0 and 1
    a = 3
    b = 5
    c = 2
    no_noise = a*(X_all)**2 + b*(X_all) + c
    y_all = no_noise + noise*rng.randn(n)
    shuffled_indices = rng.permutation(n)
    train_indices = shuffled_indices[:n_train]
    test_indices = shuffled_indices[n_train:]
    
    # Now break up the two inputs and outputs
    X_train = X_all[train_indices]
    y_train = y_all[train_indices]
    X_test = X_all[test_indices]
    y_test = y_all[test_indices]
    return (X_train, y_train, X_test, y_test)

def load_california(n_train: int=100, n_test: int=20, seed: int=42):
    pass

def make_synth_clf_linear(n_train: int=100, n_test: int=20, noise: float=0.1, seed: int=42):
    pass

def load_breast_cancer(n_train: int=100, n_test: int=20, seed: int=42):
    pass

def load_digits_full(n_train: int=100, n_test: int=20, seed: int=42):
    pass

def load_digits_4v9(n_train: int=100, n_test: int=20, seed: int=42):
    pass