import numpy as np

def gd_linear_step(X: np.ndarray, y: np.ndarray, w: np.ndarray, eta: float, l1:float=0.0, l2:float=0.0):
    """Compute gradient of linear regression loss given inputs, expected outputs, and other parameters

    Args:
        X (np.ndarray): inputs
        y (np.ndarray): expected outputs
        w (np.ndarray): weights of linear regression model
        eta (float): learning rate
        l1 (float, optional): l1 regularization constant for weights. Defaults to 0.0.
        l2 (float, optional): l2 regularization constant for weights. Defaults to 0.0.
    """
    n = len(X)
    predictions = np.matmul(X, y)
    residuals = y - predictions
    # Formula for the gradiant (taking away the scalar of 1/2 because it is just a positive scalar - if you minimuze half the loss then you minimize twice the loss)
    grad = np.matmul(X.T, residuals) / n
    # Add regularization factors to loss
    grad += l2*w # The addition to the loss was half the square of the magnitude of the weights
    
    pass

def gd_logistic_step(X: np.ndarray, y: np.ndarray, w: np.ndarray, eta: float, l1:float=0.0, l2:float=0.0):
    """Compute gradient of logistic regression loss given inputs, expected outputs, and other parameters

    Args:
        X (np.ndarray): inputs  
        y (np.ndarray): expected outputs
        w (np.ndarray): weights of logistic regression model
        eta (float): dunno...
        l1 (float, optional): regularization constant for weights. Defaults to 0.0.
        l2 (float, optional): regularization constant for weights squared. Defaults to 0.0.
    """