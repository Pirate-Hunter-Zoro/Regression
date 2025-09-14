import numpy as np
import unittest
from src.utils import *

class TestUtils(unittest.TestCase):
    
    def test_add_bias_2d(self):
        old_X = np.array([[2,3],[4,5],[6,7]])
        X = add_bias(old_X)
        # 1's should have been added as a column
        self.assertTrue(X.shape == (3,3))
        # Make sure they are ones
        self.assertTrue(np.allclose(X[:,0], 1.0))
        # Make sure nothing else changed
        self.assertTrue(np.allclose(X[:,1:], old_X)) # all rows, exclude first column because those are the ones
        
    def test_standardize(self):
        X = np.array([[1,2],[3,4]])
        X, mu, sigma = standardize(X)
        self.assertTrue(np.allclose(mu, np.array([2,3])))
        self.assertTrue(np.allclose(sigma, np.array([1,1])))
        self.assertTrue(np.allclose(X, np.array([[-1,-1],[1,1]])))
        
        X = np.array([5,5,5])
        X, mu, sigma = standardize(X)
        self.assertTrue(np.allclose(mu, np.array([5])))
        self.assertTrue(np.allclose(sigma, np.array([1])))
        self.assertTrue(np.allclose(X, np.array([[0],[0],[0]])))
        
    def test_train_test_sample(self):
        X = np.array([1,2,3,4,5,6,7,8,9,10])
        y = np.array([2,4,6,8,10,12,14,16,18,20])
        n_train = 3
        n_test = 4
        X_train, y_train, X_test, y_test = train_test_sample(X, y, n_train, n_test)
        self.assertTrue(X_train.shape == (3,1) and y_train.shape == (3,) and X_test.shape == (4,1) and y_test.shape == (4,))
        new_X_train, new_y_train, new_X_test, new_y_test = train_test_sample(X, y, n_train, n_test)
        self.assertTrue(np.allclose(new_X_train, X_train) and np.allclose(new_y_train, y_train) and np.allclose(new_X_test, X_test) and np.allclose(new_y_test, y_test))
        
    def test_soft_threshold(self):
        w = np.array([3.0, -2.0, -.5])
        alpha = 1.0
        self.assertTrue(np.allclose(soft_threshold(w, alpha), np.array([2.0,-1.0,0.0])))
        w = np.array([0.0, 2.0])
        alpha = 1.0
        self.assertTrue(np.allclose(soft_threshold(w, alpha), np.array([0.0, 1.0])))
        w = np.array([0.2,-0.2])
        alpha = 0.5
        self.assertTrue(np.allclose(soft_threshold(w, alpha), np.array([0,0])))