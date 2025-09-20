from src.datasets import *
from src.cv import *
from src.models_linear import *
from src.models_logistic import *
import unittest
import numpy as np

class TestDatasets(unittest.TestCase):
    
    def test_make_synth_reg_linear(self):   
        n_train = 20
        n_test = 80
        X_train, y_train, X_test, y_test = make_synth_reg_linear(n_train, n_test)
        self.assertTrue(X_train.shape == (n_train,))
        self.assertTrue(y_train.shape == (n_train,))
        self.assertTrue(X_test.shape == (n_test,))
        self.assertTrue(y_test.shape == (n_test,))
        new_X_train, new_y_train, new_X_test, new_y_test = make_synth_reg_linear(n_train, n_test)
        self.assertTrue(np.array_equal(X_train, new_X_train) and np.array_equal(X_test, new_X_test) and np.array_equal(y_train, new_y_train) and np.array_equal(y_test, new_y_test))
        
    def test_make_synth_reg_quadratic(self):
        n_train = 20
        n_test = 80
        X_train, y_train, X_test, y_test = make_synth_reg_quadratic(n_train, n_test)
        self.assertTrue(X_train.shape == (n_train,))
        self.assertTrue(y_train.shape == (n_train,))
        self.assertTrue(X_test.shape == (n_test,))
        self.assertTrue(y_test.shape == (n_test,))
        new_X_train, new_y_train, new_X_test, new_y_test = make_synth_reg_quadratic(n_train, n_test)
        self.assertTrue(np.array_equal(X_train, new_X_train) and np.array_equal(X_test, new_X_test) and np.array_equal(y_train, new_y_train) and np.array_equal(y_test, new_y_test))
        # Also test to make sure the data is quadratic
        
        _, best, _ = cv_grid_search(model_fn=fit_gd_linear,predict_fn=predict_linear, X=X_train, y=y_train)
        best_mse = best["mean_metric"]
        
        # Now square X_train, and see if we can get a better fit out of that (we should be able to)
        X_train_squared = np.zeros((n_train, 2))
        X_train_squared[:,0] = X_train**2 # Find the a for ax
        X_train_squared[:,1] = X_train # Find the b for bx
        # We just linearized the problem
        _, new_best, _ = cv_grid_search(model_fn=fit_gd_linear, predict_fn=predict_linear, X=X_train_squared, y=y_train)
        new_best_mse = new_best["mean_metric"]
        
        self.assertTrue(abs(new_best_mse) < abs(best_mse))
        
    def test_make_synth_clf(self):
        n_train = 20
        n_test = 80
        X_train, y_train, X_test, y_test = make_synth_clf(n_train, n_test)
        self.assertTrue(X_train.shape == (n_train,2))
        self.assertTrue(y_train.shape == (n_train,))
        self.assertTrue(X_test.shape == (n_test,2))
        self.assertTrue(y_test.shape == (n_test,))
        new_X_train, new_y_train, new_X_test, new_y_test = make_synth_clf(n_train, n_test)
        self.assertTrue(np.array_equal(X_train, new_X_train) and np.array_equal(X_test, new_X_test) and np.array_equal(y_train, new_y_train) and np.array_equal(y_test, new_y_test))
        
        # Test to make sure we did well with classification
        _, best, _ = cv_grid_search(model_fn=fit_gd_logistic,predict_fn=predict_labels_softmax, X=X_train, y=y_train, param_grid={"iters":[300]}, task="clf")
        best_accuracy = best["mean_metric"]
        self.assertTrue(best_accuracy > 0.9)
        
    def test_load_california(self):
        n_train = 200
        n_test = 300
        seed = 123
        X_train, y_train, X_test, y_test = load_california(n_train, n_test, seed)
        self.assertTrue(X_train.shape == (n_train, 8))
        self.assertTrue(X_test.shape == (n_test, 8))
        self.assertTrue(y_train.shape == (n_train,))
        self.assertTrue(y_test.shape == (n_test,))
        
        new_X_train, new_y_train, new_X_test, new_y_test = load_california(n_train, n_test, seed)
        self.assertTrue(np.array_equal(X_train, new_X_train) and np.array_equal(y_train, new_y_train) and np.array_equal(X_test, new_X_test) and np.array_equal(y_test, new_y_test))
        
        _, best, _ = cv_grid_search(model_fn=fit_gd_linear,predict_fn=predict_linear, X=X_train, y=y_train)
        self.assertTrue(np.isfinite(best["mean_metric"])) # An infinite mean squared error would be... troubling
        self.assertTrue(best["mean_metric"] > 0) # But we will still have SOME error
        
    def test_load_breast_cancer_data(self):
        n_train = 200
        n_test = 300
        d = 30
        seed = 123
        X_train, y_train, X_test, y_test = load_breast_cancer_data(n_train, n_test, seed)
        self.assertTrue(X_train.shape == (n_train, d))
        self.assertTrue(X_test.shape == (n_test, d))
        self.assertTrue(y_train.shape == (n_train,))
        self.assertTrue(y_test.shape == (n_test,))
        
        new_X_train, new_y_train, new_X_test, new_y_test = load_breast_cancer_data(n_train, n_test, seed)
        self.assertTrue(np.array_equal(X_train, new_X_train) and np.array_equal(y_train, new_y_train) and np.array_equal(X_test, new_X_test) and np.array_equal(y_test, new_y_test))
        
        _, best, _ = cv_grid_search(model_fn=fit_gd_logistic,predict_fn=predict_labels_softmax, X=X_train, y=y_train, task="clf")
        self.assertTrue(best["mean_metric"] < 1) # A greater than 1 accuracy makes no sense
        self.assertTrue(best["mean_metric"] > 0.9) # The data set is "easy" so we'd better be classifying somewhat well...
        
    def test_load_digits_full(self):
        n_train = 200
        n_test = 300
        d = 64
        seed = 123
        X_train, y_train, X_test, y_test = load_digits_full(n_train, n_test, seed)
        self.assertTrue(X_train.shape == (n_train, d))
        self.assertTrue(X_test.shape == (n_test, d))
        self.assertTrue(y_train.shape == (n_train,))
        self.assertTrue(y_test.shape == (n_test,))
        
        new_X_train, new_y_train, new_X_test, new_y_test = load_digits_full(n_train, n_test, seed)
        self.assertTrue(np.array_equal(X_train, new_X_train) and np.array_equal(y_train, new_y_train) and np.array_equal(X_test, new_X_test) and np.array_equal(y_test, new_y_test))
        
        _, best, _ = cv_grid_search(model_fn=fit_gd_logistic,predict_fn=predict_labels_softmax, X=X_train, y=y_train, task="clf")
        self.assertTrue(best["mean_metric"] < 1) # A greater than 1 accuracy makes no sense
        self.assertTrue(best["mean_metric"] > 0.8) # A reasonable performance expectation...
        
    def test_load_digits_4v9(self):
        n_train = 50
        n_test = 100
        d = 64
        seed = 123
        X_train, y_train, X_test, y_test = load_digits_4v9(n_train, n_test, seed)
        self.assertTrue(set(np.unique(y_train)) == {0,1})
        self.assertTrue(X_train.shape == (n_train, d))
        self.assertTrue(X_test.shape == (n_test, d))
        self.assertTrue(y_train.shape == (n_train,))
        self.assertTrue(y_test.shape == (n_test,))
        
        new_X_train, new_y_train, new_X_test, new_y_test = load_digits_4v9(n_train, n_test, seed)
        self.assertTrue(np.array_equal(X_train, new_X_train) and np.array_equal(y_train, new_y_train) and np.array_equal(X_test, new_X_test) and np.array_equal(y_test, new_y_test))
        
        _, best, _ = cv_grid_search(model_fn=fit_gd_logistic,predict_fn=predict_labels_softmax, X=X_train, y=y_train, task="clf")
        self.assertTrue(best["mean_metric"] < 1) # A greater than 1 accuracy makes no sense
        self.assertTrue(best["mean_metric"] > 0.95) # The binary case for digits should be even easier...