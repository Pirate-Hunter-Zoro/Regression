import numpy as np
import unittest
from src.utils import *

class TestUtils(unittest.TestCase):
    
    def test_add_bias_2d(self):
        old_X = np.array([[2,3],[4,5],[6,7]])
        X = add_bias(old_X)
        # 1's should have been added as a column
        assert X.shape == (3,3)
        # Make sure they are ones
        assert np.allclose(X[:,0], 1.0)
        # Make sure nothing else changed
        assert np.allclose(X[:,1:], old_X) # all rows, exclude first column because those are the ones
        
    def test_standardize(self):
        X = np.array([[1,2],[3,4]])
        X, mu, sigma = standardize(X)
        assert np.allclose(mu, np.array([2,3]))
        assert np.allclose(sigma, np.array([1,1]))
        assert np.allclose(X, np.array([[-1,-1],[1,1]]))
        
        X = np.array([5,5,5])
        X, mu, sigma = standardize(X)
        assert np.allclose(mu, np.array([5]))
        assert np.allclose(sigma, np.array([1]))
        assert np.allclose(X, np.array([[0],[0],[0]]))