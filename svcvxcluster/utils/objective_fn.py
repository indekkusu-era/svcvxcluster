import numpy as np

def obj_primal(X, B, Z, A, eps, C, mu, **kwargs):
    return 0.5 * np.linalg.norm(X - A) ** 2 + C * np.sum(np.clip(np.abs(X @ B) - eps, 0, np.inf))

def obj_dual(X, B, Z, A, eps, C, mu, **kwargs):
    normsqA = np.linalg.norm(A) ** 2
    return - (0.5 * np.linalg.norm(X) ** 2 + eps * np.sum(np.abs(Z)) - normsqA * 0.5)
