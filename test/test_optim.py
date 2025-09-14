import numpy as np
import unittest
from src.optim import *

class TestOptim(unittest.TestCase):
    
    def test_gd_linear_step(self):
        X = np.array([[1,1],[1,2]])
        y = np.array([2,3])
        w = np.array([0,0])
        
        w_new = gd_linear_step(X, y, w, eta=0.1)
        self.assertTrue(np.allclose(w_new, np.array([0.25,0.4])))
        
    def test_gd_softmax_step(self):
        X = np.array([[1,0],[1,1]])
        y = np.array([0,1])
        k = 3
        w = np.zeros(shape=(len(X[0]),k))
        
        w_new = gd_softmax_step(X, y, k, w, eta=1)
        grads_expected = np.array([[-1/6,-1/6,1/3],[1/6,-1/3,1/6]])
        w_expected = w - grads_expected
        self.assertTrue(np.allclose(w_new, w_expected))
        
        