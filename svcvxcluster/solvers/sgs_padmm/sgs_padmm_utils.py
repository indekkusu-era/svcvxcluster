import numpy as np

def prox(X, eps, C, mu):
    return np.sign(X) * np.clip(np.abs(X) - mu * eps, 0, C)
