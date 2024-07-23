import numpy as np
from .admm_utils import obj_function

def cg(Z, laplacian_mat, incidence_mat, mu, n_iter=100, X0=None, tol=1e-5):
    if X0 is None:
        X = Z.copy()
        R = 2 * mu * laplacian_mat @ X
    else:
        X = X0.copy()
        R = X + 2 * mu * laplacian_mat @ X - Z
    P = - R
    normR = np.linalg.norm(R, axis=0)
    for i in range(n_iter):
        AP = P + 2 * mu * laplacian_mat @ P
        alpha = - (R * P).sum(axis=0) / (P * AP).sum(axis=0)
        X += alpha * P
        R += alpha * AP
        beta = ((normR2 := np.linalg.norm(R, axis=0)) / normR) ** 2
        P = - R + beta * P
        if np.max(normR2) < tol:
            break
        normR = normR2
    return X, i + 1

def armijo_line_search(P, L, A, mu, eps, C, dP, gradP, inc_mat, alpha0=1, beta=0.5, sigma=1, max_iter=10):
    alpha = alpha0 / beta
    fp = obj_function(P, L, A, mu, eps, C, inc_mat)
    grad_dot = np.sum(gradP * dP)
    for i in range(max_iter):
        alpha *= beta
        if obj_function(P + alpha * dP, L, A, mu, eps, C, inc_mat) <= fp + alpha * sigma * grad_dot:
            break
    return alpha
