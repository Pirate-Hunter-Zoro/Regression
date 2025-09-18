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

def make_synth_clf(n_train: int=100, n_test: int=20, noise: float=0.1, seed: int=42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
    """Generate synthetic linear classification data with the specified number of training and testing cases and noise

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
    center = 1
    left_mean = np.array([-center,0])
    right_mean = np.array([center,0])
    # "Isotropic covariance" - circular regions, so features are not correlated
    covariance = noise*noise*np.eye(2) # for BOTH clusters
    class_1_count = n // 2
    class_2_count = n - class_1_count
    
    # Sample from both classes with their respective multivariate normal distributions
    class_1_points = rng.multivariate_normal(left_mean,covariance,size=class_1_count)
    class_2_points = rng.multivariate_normal(right_mean,covariance,size=class_2_count)
    
    X = np.concatenate([class_1_points, class_2_points], axis=0) # Shove the two datasets together
    y = np.ones(n, dtype=int)
    y[:class_1_count] = 0
    
    permuted_indices = rng.permutation(n)
    X_permuted = X[permuted_indices]
    y_permuted = y[permuted_indices]
    
    train_X = X_permuted[:n_train]
    test_X = X_permuted[n_train:]
    train_y = y_permuted[:n_train]
    test_y = y_permuted[n_train:]
    return train_X, train_y, test_X, test_y

def load_california(n_train: int=100, n_test: int=20, seed: int=42):
    pass

def load_breast_cancer(n_train: int=100, n_test: int=20, seed: int=42):
    pass

def load_digits_full(n_train: int=100, n_test: int=20, seed: int=42):
    pass

def load_digits_4v9(n_train: int=100, n_test: int=20, seed: int=42):
    pass