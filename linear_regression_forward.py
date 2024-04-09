import numpy as np
from numpy.typing import NDArray

def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
    # X is an Nx3 NumPy array
    # weights is a 3x1 NumPy array
    # HINT: np.matmul() will be useful
    # return np.round(your_answer, 5)
    return np.round(X @ weights, 5)

def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
    # model_prediction is an Nx1 NumPy array
    # ground_truth is an Nx1 NumPy array
    # HINT: np.mean(), np.square() will be useful
    # return round(your_answer, 5)
    return np.round(np.mean((model_prediction - ground_truth)**2), 5)
