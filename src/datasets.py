import numpy as np
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_digits

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

def load_california(n_train: int=100, n_test: int=20, seed: int=42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load california housing linear regression data

    Args:
        n_train (int, optional): Number of training observations. Defaults to 100.
        n_test (int, optional): Number of testing observations. Defaults to 20.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, y_train, X_test, y_test
    """
    data = fetch_california_housing(as_frame=False) # We want numpy array
    X_all = data.data
    y_all = data.target
    if not isinstance(X_all, np.ndarray) or not isinstance(y_all, np.ndarray):
        raise ValueError(f"Must convert California housing data into numpy arrays - currently they are {X_all} and {y_all}")
    n = X_all.shape[0]
    if not (n_train + n_test <= n and n_train > 0 and n_test > 0):
        raise ValueError(f"Received invalid train and testing sizes of {n_train} and {n_test} respectivly")
    if len(X_all.shape) != 2:
        raise ValueError(f"Must turn housing data into 2D array")
    if X_all.shape[1] != 8:
        raise ValueError(f"Expected 8 features in the housing observations...")
    if len(y_all.shape) > 2:
        raise ValueError(f"Expected 1D or 2D output shape but recieved {y_all.shape}")
    if y_all.shape[0] != n:
        raise ValueError(f"Expected y to have the same count as X ({n}) but received {y_all.shape[0]}")
    if y_all.shape == (n,1):
        y_all = y_all.reshape((n,))
    
    rng = np.random.RandomState(seed)
    permuted_indices = rng.permutation(n)
    X_sample = X_all[permuted_indices[:n_train+n_test]]
    y_sample = y_all[permuted_indices[:n_train+n_test]]
    X_train = X_sample[:n_train]
    X_test = X_sample[n_train:]
    y_train = y_sample[:n_train]
    y_test = y_sample[n_train:]
    
    if (not np.issubdtype(X_train.dtype, np.floating)) or (not np.issubdtype(y_train.dtype, np.floating)):
        raise ValueError(f"Must have float type arrays")
    
    return X_train, y_train, X_test, y_test

def load_breast_cancer_data(n_train: int=100, n_test: int=20, seed: int=42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load breast cancer logistic regression data

    Args:
        n_train (int, optional): Number of training observations. Defaults to 100.
        n_test (int, optional): Number of testing observations. Defaults to 20.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, y_train, X_test, y_test
    """
    data = load_breast_cancer(as_frame=False) # Again we want numpy arrays
    X_all = data.data
    y_all = data.target
    n = X_all.shape[0]
    d = X_all.shape[1] # number of features
    if not X_all.shape == (n, d):
        raise ValueError(f"Unexpected loaded data of shape {X_all.shape} when expecting {(n,d)}...")
    if y_all.shape == (n,1):
        y_all = y_all.reshape((n,))
    if not y_all.shape == (n,):
        raise ValueError(f"Unexpected loaded target of shape {y_all.shape} when expecting {(n,)}...")
    if not np.issubdtype(y_all.dtype, np.integer):
        raise ValueError(f"Unexpected type for target - got {y_all.dtype}...")
    if not np.issubdtype(X_all.dtype, np.floating):
        raise ValueError(f"Unexpected type for data - got {X_all.dtype}...")
    
    if not (n_train + n_test <= n and n_train > 0 and n_test > 0):
        raise ValueError(f"Received invalid train and testing sizes of {n_train} and {n_test} respectivly")
    
    rng = np.random.RandomState(seed)
    permuted_indices = rng.permutation(n)
    X_sample = X_all[permuted_indices[:n_train+n_test]]
    y_sample = y_all[permuted_indices[:n_train+n_test]]
    X_train = X_sample[:n_train]
    X_test = X_sample[n_train:]
    y_train = y_sample[:n_train]
    y_test = y_sample[n_train:]
    
    return X_train, y_train, X_test, y_test
    
def load_digits_full(n_train: int=100, n_test: int=20, seed: int=42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load digits for logistic regression data

    Args:
        n_train (int, optional): Number of training observations. Defaults to 100.
        n_test (int, optional): Number of testing observations. Defaults to 20.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, y_train, X_test, y_test
    """
    data = load_digits(as_frame=False) # Again we want numpy arrays
    X_all = data.data
    y_all = data.target
    n = X_all.shape[0]
    d = X_all.shape[1] # number of features
    if not X_all.shape == (n, d):
        raise ValueError(f"Unexpected loaded data of shape {X_all.shape} when expecting {(n,d)}...")
    if y_all.shape == (n,1):
        y_all = y_all.reshape((n,))
    if not y_all.shape == (n,):
        raise ValueError(f"Unexpected loaded target of shape {y_all.shape} when expecting {(n,)}...")
    if not np.issubdtype(y_all.dtype, np.integer):
        raise ValueError(f"Unexpected type for target - got {y_all.dtype}...")
    if not np.issubdtype(X_all.dtype, np.floating):
        raise ValueError(f"Unexpected type for data - got {X_all.dtype}...")
    
    if not (n_train + n_test <= n and n_train > 0 and n_test > 0):
        raise ValueError(f"Received invalid train and testing sizes of {n_train} and {n_test} respectivly")
    
    rng = np.random.RandomState(seed)
    permuted_indices = rng.permutation(n)
    X_sample = X_all[permuted_indices[:n_train+n_test]]
    y_sample = y_all[permuted_indices[:n_train+n_test]]
    X_train = X_sample[:n_train]
    X_test = X_sample[n_train:]
    y_train = y_sample[:n_train]
    y_test = y_sample[n_train:]
    
    return X_train, y_train, X_test, y_test

def load_digits_4v9(n_train: int=100, n_test: int=20, seed: int=42):
    """Load ONLY 4 and 9 digits for logistic regression data

    Args:
        n_train (int, optional): Number of training observations. Defaults to 100.
        n_test (int, optional): Number of testing observations. Defaults to 20.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, y_train, X_test, y_test
    """
    data = load_digits(as_frame=False) # Again we want numpy arrays
    X_all = data.data
    y_all = data.target
    n = X_all.shape[0]
    d = X_all.shape[1] # number of features
    if not X_all.shape == (n, d):
        raise ValueError(f"Unexpected loaded data of shape {X_all.shape} when expecting {(n,d)}...")
    if y_all.shape == (n,1):
        y_all = y_all.reshape((n,))
    if not y_all.shape == (n,):
        raise ValueError(f"Unexpected loaded target of shape {y_all.shape} when expecting {(n,)}...")
    if not np.issubdtype(y_all.dtype, np.integer):
        raise ValueError(f"Unexpected type for target - got {y_all.dtype}...")
    if not np.issubdtype(X_all.dtype, np.floating):
        raise ValueError(f"Unexpected type for data - got {X_all.dtype}...")
    
    # Now we filter for only 4 and 9
    X_all = X_all[(y_all==4) | (y_all==9)]
    y_all = y_all[(y_all==4) | (y_all==9)]
    # Remap to binary classification
    y_all[(y_all==4)] = 0
    y_all[(y_all==9)] = 1
    n = X_all.shape[0]
    
    if not (n_train + n_test <= n and n_train > 0 and n_test > 0):
        raise ValueError(f"Received invalid train and testing sizes of {n_train} and {n_test} respectivly")
    
    rng = np.random.RandomState(seed)
    permuted_indices = rng.permutation(n)
    X_sample = X_all[permuted_indices[:n_train+n_test]]
    y_sample = y_all[permuted_indices[:n_train+n_test]]
    X_train = X_sample[:n_train]
    X_test = X_sample[n_train:]
    y_train = y_sample[:n_train]
    y_test = y_sample[n_train:]
    
    return X_train, y_train, X_test, y_test