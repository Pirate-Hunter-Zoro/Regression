import unittest
import numpy as np
from src.models_logistic import *

class TestModelLogistic(unittest.TestCase):
    
    def test_model(self):
        
        X = np.array([[1,0],[1,1]])
        y = np.array([0,1])
        K = 3
        
        eta = 0.1
        iters = 50
        l1=0
        l2=0
        
        W = fit_gd_logistic(X, y, K, eta, iters, l1, l2, verbose=True)
        self.assertTrue(W.shape == (2,3))
        
        Z = predict_labels_softmax(X, W)
        self.assertTrue(Z.shape == (2,))
        # The model had BETTER be correct with this small a problem after 50 iterations
        self.assertTrue(Z[0] == 0)
        self.assertTrue(Z[1] == 1)