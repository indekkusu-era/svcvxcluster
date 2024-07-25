import numpy as np
from scipy.sparse import csr_array, lil_array

def prox(X, eps, C, mu):
    return np.sign(X) * np.clip(np.abs(X) - mu * eps, 0, C)

def dprox(X, eps, C, mu, flatten=False):
    xbar = np.abs(X) - mu * eps
    if flatten:
        return lil_array(((xbar >= 0) & (xbar <= C)).astype(int))
    return csr_array(((xbar >= 0) & (xbar <= C)).astype(int))
 
def ssnal_grad(X, A, B, prox):
    return X - A + prox @ B.T

def obj_function(X, A, Z, B, eps, C, mu):
    bx = X @ B
    prox_var = bx / mu + Z
    proxU = prox(prox_var, eps, C, 1 / mu)
    U = mu * (prox_var - proxU)
    moreau = C * np.sum(np.clip(np.abs(U) - eps, 0, np.inf)) + mu * np.linalg.norm(proxU) ** 2 / 2
    return 0.5 * np.linalg.norm(X - A) ** 2 + moreau - mu * np.linalg.norm(Z) ** 2 / 2
