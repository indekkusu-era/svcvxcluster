import numpy as np
from scipy.sparse import identity as spI
from scipy.sparse import linalg as splinalg
from .ssnal_utils import obj_function

def ssnal_cg(B, grad, mu, n, subgrad_P, X0=None, tol=1e-5, parallel=False, d=None):
    # solve Hx + g = 0
    if X0 is None:
        dX = - grad.copy()
    else:
        dX = X0.copy()
    if not parallel:
        for i, row in enumerate(subgrad_P):
            BP = B * row
            sub_laplacian = BP @ BP.T
            mat = spI(n) + sub_laplacian / mu
            dX[i], _ = splinalg.cg(mat, -grad[i], atol=tol, x0=dX[i])
    else:
        raise NotImplementedError("Parallel Implementation is currently not supported")

    return dX

def armijo_line_search(X, A, Z, B, eps, C, mu, grad, d, alpha0=1, beta=0.5, sigma=0.5, max_iter=10):
    alpha = alpha0 / beta
    fp = obj_function(X, A, Z, B, eps, C, mu)
    grad_dot = np.sum(grad * d)
    fprev = np.inf
    for i in range(max_iter):
        alpha *= beta
        fxd = obj_function(X + alpha * d, A, Z, B, eps, C, mu)
        if fxd <= fp + alpha * sigma * grad_dot:
            break
        if fxd > fprev:
            alpha /= beta
            break
        fprev = fxd
    return alpha

def armijo_dual(X, A, Z, B, eps, C, mu, grad, d, alpha0=1, beta=0.5, sigma=0.5, max_iter=10):
    alpha = alpha0 / beta
    fp = obj_function(X, A, Z, B, eps, C, mu)
    grad_dot = np.sum(grad * d)
    fprev = -np.inf
    for i in range(max_iter):
        alpha *= beta
        fxd = obj_function(X, A, Z + alpha * d, B, eps, C, mu)
        if fxd >= fp + alpha * sigma * grad_dot:
            break
        if fxd < fprev:
            alpha /= beta
            break
        fprev = fxd
    return alpha

