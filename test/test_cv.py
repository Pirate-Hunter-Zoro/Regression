import numpy as np
import unittest
from src.cv import *

class TestCV(unittest.TestCase):
    
    def test_kfold_indices(self):
        n = 10
        k = 3
        seed = 123
        
        folds = kfold_indices(n, k, seed)
        self.assertTrue(len(folds) == k)
        all_indices = np.arange(n)
        for split in folds:
            self.assertTrue(isinstance(split[0], np.ndarray))
            self.assertTrue(split[0].dtype == int)
            self.assertTrue(isinstance(split[1], np.ndarray))
            self.assertTrue(split[1].dtype == int)
            self.assertTrue(len(split[0]) + len(split[1]) == n)
            self.assertTrue(len(split[1]) == n//k or len(split[1]) == n//k+1)
            
            fold_indices = np.concatenate([split[0],split[1]])
            fold_indices = np.sort(fold_indices)
            self.assertTrue(np.array_equal(fold_indices, all_indices))
            
        new_folds = kfold_indices(n, k, seed)
        # Ensure same random seed yields deterministic result
        for fold1, fold2 in zip(folds, new_folds):
            self.assertTrue(np.array_equal(fold1[0],fold2[0]) and np.array_equal(fold1[1],fold2[1]))