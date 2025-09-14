import numpy as np

def kfold_indices(n: int, k:int=10, seed:int=42) -> list[tuple[int,int]]:
    pass

def cv_grid_search(model_fn, predict_fn, X:np.ndarray, y:np.ndarray, k:int, param_grid, task):
    pass