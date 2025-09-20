import numpy as np
from src.optim import *

def fit_normal_eq_ridge(X: np.ndarray, y: np.ndarray, l2: float=0) -> np.ndarray:
    """Directly solve for the weights that optimize MSE loss via linear algebra

    Args:
        X (np.ndarray): input array
        y (np.ndarray): expected output values
        l2 (float, optional): Regularization constant. Defaults to 0.

    Returns:
        np.ndarray: resulting weights which optimize MSE loss
    """
    if not len(X.shape) == 2:
        raise ValueError(f"Expected 2D matrix for input observations but received shape {X.shape}")
    n = X.shape[0]
    d = X.shape[1]
    if not np.allclose(X[:,0], np.ones(n)):
        raise ValueError(f"Expected input X to have first column be bias column of 1's so that the first element in the weight vector is the bias, but did not observe such...")
    if not y.shape == (n,):
        raise ValueError(f"Expected outputs to have shape {(n,)} but recieved {y.shape}")
    if l2 < 0:
        raise ValueError(f"Expected non-negative value for l2 but recieved {l2}")
    
    # Direct solve for linear regression to minimuze loss. 
    # Minimize 1/2n||Xw-y||^2 + l2/2||w_1||^2, where w_1 refers to ONLY the weight parameters (everything but the first element as described above)
    # Gradient becomes 1/nX^T(Xw-y) + l2w_1 by the chain rule
    # So solve X^T(Xw-y)+n*l2*w_1=0 <=> (X^TX + l2*n*R)w=X^Ty
    # Where 'R' is the identity matrix except for the top left element being 0 instead of 1, to avoid affecting the bias parameter
    R = np.eye(d)
    R[0][0] = 0 # l2 does not apply to bias term
    A = np.matmul(X.T,X)+l2*n*R
    b = np.matmul(X.T, y)
    # Solve for Aw=b
    w = np.linalg.solve(A, b)
    return w

def fit_gd_linear(X: np.ndarray, y: np.ndarray, eta: float, iters: int, l1: float=0.0, l2: float=0.0, tol=None, verbose: bool=False) -> np.ndarray:
    if not len(X.shape) == 2:
        raise ValueError(f"Expected 2D matrix for input observations but received shape {X.shape}")
    n = X.shape[0]
    d = X.shape[1]
    if not np.allclose(X[:,0], np.ones(n)):
        raise ValueError(f"Expected input X to have first column be bias column of 1's so that the first element in the weight vector is the bias, but did not observe such...")
    if not y.shape == (n,):
        raise ValueError(f"Expected outputs to have shape {(n,)} but recieved {y.shape}")
    if l1 < 0:
        raise ValueError(f"Expected non-negative value for l1 but recieved {l1}")
    if l2 < 0:
        raise ValueError(f"Expected non-negative value for l2 but recieved {l2}")
    if eta <= 0:
        raise ValueError(f"Expected positive learning rate but received {eta}")
    if iters < 0:
        raise ValueError(f"Expected non-negative iterations but received {iters}")

    w = np.zeros(d)
    for i in range(iters):
        w_next = gd_linear_step(X, y, w, eta, l1, l2)
        if tol is not None:
            if np.linalg.norm(w-w_next) < tol:
                break
        w = w_next
        if i % 10 == 0 and verbose:
            print(f"Data Loss: {1/(2*n)*np.linalg.norm(np.matmul(X,w)-y)**2}")
            print(f"Ridge Loss: {l2/2*np.linalg.norm(w[1:])**2}")
            print(f"Lasso Loss: {l1*np.sum(np.abs(w[1:]))}")
    
    return w

def predict_linear(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.matmul(X, w)