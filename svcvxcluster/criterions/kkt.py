import numpy as np

def prox(X, eps, C, mu):
    return np.sign(X) * np.clip(np.abs(X) - mu * eps, 0, C)

def calc_U(X, B, mu, Z, eps, C):
    BX = X @ B
    prox_var = BX / mu + Z
    U = mu * (prox_var - prox(prox_var, eps, C, 1 / mu))
    return U

def primal_kkt_residual(X, B, mu, Z, eps, C, U=None, BXDiff=None, **kwargs):
    if U is None:
        U = calc_U(X, B, mu, Z, eps, C)
    if BXDiff is None:
        BX = X @ B
        BXDiff = BX - U
    if 'normgradZ' in kwargs.keys():
        return kwargs['normgradZ']
    return np.linalg.norm(BXDiff)

def primal_relative_kkt_residual(X, B, mu, Z, eps, C, U=None, BXDiff=None, **kwargs):
    if U is None:
        U = calc_U(X, B, mu, Z, eps, C)
    if 'normU' not in kwargs.keys():
        normU = np.linalg.norm(U)
    else:
        normU = kwargs['normU']
    return primal_kkt_residual(X, B, mu, Z, eps, C, U, BXDiff, **kwargs) / (1 + normU)

def dual_kkt_residual(Z, C, **kwargs):
    return np.sum(np.clip(np.abs(Z) - C, 0, np.inf))

def dual_relative_kkt_residual(Z, C, A, **kwargs):
    if 'normA' not in kwargs.keys():
        normA = np.linalg.norm(A)
    else:
        normA = kwargs['normA']
    return dual_kkt_residual(Z, C) / (1 + normA)

def kkt_gap(X, B, Z, A, eps, C, mu, U=None, **kwargs):
    if U is None:
        U = calc_U(X, B, mu, Z, eps, C)
    return (np.linalg.norm(Z @ B.T + X - A) + np.linalg.norm(prox(U + Z, eps, C, 1) - Z))

def kkt_relative_gap(X, B, Z, A, eps, C, mu, U=None, **kwargs):
    if U is None:
        U = calc_U(X, B, mu, Z, eps, C)
    if 'normA' not in kwargs.keys():
        normA = np.linalg.norm(A)
    else:
        normA = kwargs['normA']
    if 'normU' not in kwargs.keys():
        normU = np.linalg.norm(U)
    else:
        normU = kwargs['normU']
    return kkt_gap(X, B, Z, A, eps, C, mu, U) / (1 + normA + normU)
