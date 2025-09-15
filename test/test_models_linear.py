import numpy as np
import unittest

from src.models_linear import *

class TestModelLinear(unittest.TestCase):
    
    def test_fit_normal_eq_ridge(self):
        X = np.array([[1,0],[1,1],[1,2]])
        y = np.array([1,2,3])
        w_first = fit_normal_eq_ridge(X, y)
        self.assertTrue(np.allclose(np.matmul(X,w_first), y))
        l2=0.1
        w_second = fit_normal_eq_ridge(X, y, l2)
        self.assertTrue(np.linalg.norm(w_first[1:])>np.linalg.norm(w_second[1:]))
        self.assertTrue(np.mean((np.matmul(X, w_second) - y)**2)<0.02)
        
    def test_fit_gd_linear(self):
        X = np.array([[1,0],[1,1],[1,2]])
        y = np.array([1,2,3])
        w_first = fit_normal_eq_ridge(X, y)
        w_second = fit_gd_linear(X, y, eta=0.1, iters=100)
        self.assertTrue(np.allclose(w_first, w_second, rtol=0.02*np.ones(w_first.shape)))
        y_pred = predict_linear(X, w_second)
        self.assertTrue(np.allclose(y, y_pred, rtol=0.02*np.ones(y.shape)))