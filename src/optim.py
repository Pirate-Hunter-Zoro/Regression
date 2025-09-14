import numpy as np
from src.utils import soft_threshold

def gd_linear_step(X: np.ndarray, y: np.ndarray, w: np.ndarray, eta: float, l1:float=0.0, l2:float=0.0) -> np.ndarray:
    """Compute gradient of linear regression loss given inputs, expected outputs, and other parameters

    Args:
        X (np.ndarray): inputs - note that the first column must be all ones because then the bias corresponds to the first element in w
        y (np.ndarray): expected outputs
        w (np.ndarray): weights of linear regression model
        eta (float): learning rate
        l1 (float, optional): l1 regularization constant for weights. Defaults to 0.0.
        l2 (float, optional): l2 regularization constant for weights. Defaults to 0.0.
    """
    if not len(X.shape) == 2:
        raise ValueError(f"Input array must be two-dimensional but got shape {X.shape}")
    n = X.shape[0]
    d = X.shape[1]
    if not w.shape == (d,):
        raise ValueError(f"Input weights must be of shape {(d,)}, but received {w.shape}")
    if not y.shape == (n,):
        raise ValueError(f"Input expected values should be of shape {(n,)} but got shape {y.shape}")
    if l1 < 0:
        raise ValueError(f"Received negative l1 scalar of {l1}")
    if l2 < 0:
        raise ValueError(f"Received negative l2 scalar of {l2}")
    if eta < 0:
        raise ValueError(f"Received negative learning rate value of {eta}")
    first_col = X[:, 0]
    if not np.allclose(first_col, np.ones(shape=(n,))):
        raise ValueError("First column of input observations must be all ones...")
    predictions = np.matmul(X, w)
    residuals = predictions - y
    # Formula for the gradiant (Since the scalar of 1/2 was present in the loss, it is cancelled out in the derivative)
    grad = np.matmul(X.T, residuals) / n
    # Add regularization factors to loss but ONLY the non-bias coordinate
    grad[1:] += l2*w[1:] # The addition to the loss was half the square of the magnitude of the weights
    w_new = w - eta * grad
    
    alpha = eta * l1
    # Clamp the weights corresponding with the features on scale with the product of the learning rate and l1 constants
    w_new[1:] = soft_threshold(w_new[1:], alpha)
    return w_new

def gd_softmax_step(X: np.ndarray, y: np.ndarray, K:int, w: np.ndarray, eta: float, l1:float=0.0, l2:float=0.0):
    """Compute gradient of logistic regression loss given inputs, expected outputs, and other parameters

    Args:
        X (np.ndarray): inputs - once again note that the first column must be all 1's because then the first element in w corresponds to the bias term
        y (np.ndarray): expected outputs
        k (int): number of classes
        w (np.ndarray): weights of logistic regression model
        eta (float): learning rate
        l1 (float, optional): regularization constant for weights. Defaults to 0.0.
        l2 (float, optional): regularization constant for weights squared. Defaults to 0.0.
    """
    if not len(X.shape) == 2:
        raise ValueError(f"Input array must be two-dimensional but got shape {X.shape}")
    n = X.shape[0]
    d = X.shape[1]
    if not w.shape == (d, K):
        raise ValueError(f"Input weights must be of shape {(d,K)}, but received {w.shape}")
    if not y.shape == (n,):
        raise ValueError(f"Input expected values should be of shape {(n,)} but got shape {y.shape}")
    if (not (y.dtype == int or y.dtype == np.int_)) or y.max() >= K or y.min() < 0:
            raise ValueError(f"Inputs should be positive integer values to denote classes {0} through {K-1}, but got {y}")
    if l1 < 0:
        raise ValueError(f"Received negative l1 scalar of {l1}")
    if l2 < 0:
        raise ValueError(f"Received negative l2 scalar of {l2}")
    if eta < 0:
        raise ValueError(f"Received negative learning rate value of {eta}")
    first_col = X[:, 0]
    if not np.allclose(first_col, np.ones(shape=(n,))):
        raise ValueError("First column of input observations must be all ones...")
    
    z = np.matmul(X, w)
    # Because we're about to softmax this, we can do some monotonic scaling to prevent overflow
    z = z - z.max(axis=1, keepdims=True)
    
    E = np.exp(z)
    p = E / E.sum(axis=1, keepdims=True)
    
    targets = np.eye(K)[y] # one-hot representation of shape n x K
    
    grad = np.matmul(X.T, p-targets) / n # Formula for multiclass logistic loss - the closer the matching index's probability is to 1 and the closer all others are to 0, the more accurate our model is 
    
    # Add penalty for large weights associated with the FEATURES (not bias)
    grad[1:] += l2*w[1:]
    
    w_new = w - eta*grad
    
    alpha = eta * l1
    w_new[1:] = soft_threshold(w_new[1:], alpha)
    
    return w_new