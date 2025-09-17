import numpy as np
import unittest
from src.cv import *
from src.models_linear import *
from src.models_logistic import *

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
            
    def test_cv_grid_search(self):
        n = 100
        X = np.arange(n, dtype=float)
        y_linear = 2*X+0.1*np.random.rand(n)
        y_logistic = np.zeros(n, dtype=int)
        y_logistic[X>50]=1
        
        # Test linear regression
        results, best, worst = cv_grid_search(model_fn=fit_gd_linear, predict_fn=predict_linear, X=X, y=y_linear, param_grid={"eta": [0.1, 0.01, 0.001]})
        self.assertTrue(len(results) == 3)
        self.assertTrue(best["mean_metric"] <= worst["mean_metric"])
        
        new_results, new_best,  new_worst = cv_grid_search(model_fn=fit_gd_linear, predict_fn=predict_linear, X=X, y=y_linear, param_grid={"eta": [0.1,0.01,0.001]})
        self.assertTrue(new_results == results and new_best == best and new_worst == worst)
        
        scaled_X = 1000*X
        _,scaled_best,_ = cv_grid_search(model_fn=fit_gd_linear,predict_fn=predict_linear,X=scaled_X,y=y_linear,param_grid={"eta":[0.1,0.01,0.001]})
        self.assertTrue(abs(best["mean_metric"]-scaled_best["mean_metric"]) <= 1e-10)
        
        # Test logistic regression
        results, best, worst = cv_grid_search(model_fn=fit_gd_logistic, predict_fn=predict_labels_softmax, X=X, y=y_logistic, param_grid={"eta": [0.1, 0.01]}, task="clf")
        self.assertTrue(best["mean_metric"] >= worst["mean_metric"])
        self.assertTrue(abs(best["mean_metric"] - 1) < 0.05)