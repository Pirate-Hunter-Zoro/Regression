import numpy as np
from src.optim import *

def fit_gd_logistic(X: np.ndarray, y: np.ndarray, K: int, eta: float, iters: int, l1: float=0.0, l2: float=0.0, verbose: bool=False):
    """Fit a logistic regression model on a collection of inputs and outputs

    Args:
        X (np.ndarray): input data
        y (np.ndarray): expected classifications for data
        K (int): number of classes
        eta (float): learning rate
        iters (int): number of iterations for gradient descent
        l1 (float, optional): l1 normalization constant. Defaults to 0.0.
        l2 (float, optional): l2 normalization constant. Defaults to 0.0.
        verbose (bool, optional): whether recording occurs - defaults to false
    """
    if not len(X.shape)==2:
        raise ValueError(f"Expected input X of shape (n,d) but received one of size {X.shape}")
    n = X.shape[0] # number of observations - first column had better be 1's
    d = X.shape[1] # number of features
    if y.shape != (n,):
        raise ValueError(f"Expected input y to be of shape {(n,)} but received shape {y.shape}")

    W = np.zeros(shape=(d,K))
    # Training loop
    for i in range(iters):
        W_new = gd_softmax_step(X, y, K, W, eta, l1, l2)
        W = W_new
        if i % 10 == 0 and verbose:
            Z = np.matmul(X, W)
            Z = Z - Z.max(axis=1, keepdims=True)
            E = np.exp(Z)
            P = E / E.sum(axis=1, keepdims=True)
            P = np.clip(P, 1E-12, 1-1E-12)
            Y = np.eye(K)[y]
            loss = -np.mean(np.sum(Y * np.log(P), axis=1)) + (l2/2)*np.sum(W[1:, :]**2) + l1*np.sum(np.abs(W[1:,:]))
            print(f"Iteration {i}; Loss {loss}")
    return W

def predict_scores_softmax(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.matmul(X, W)

def predict_labels_softmax(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    Z = predict_scores_softmax(X, W)
    return Z.argmax(axis=1)