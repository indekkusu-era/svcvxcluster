import numpy as np

def prox(X, eps, C, mu):
    return np.sign(X) * np.clip(np.abs(X) - mu * eps, 0, C)

def calc_U(X, B, mu, Z, eps, C):
    BX = X @ B
    prox_var = BX / mu + Z
    U = mu * (prox_var - prox(prox_var, eps, C, 1 / mu))
    return U

def primal_kkt_residual(X, B, mu, Z, eps, C, **kwargs):
    BX = X @ B
    U = calc_U(X, B, mu, Z, eps, C)
    BXDiff = BX - U
    return np.linalg.norm(BXDiff)

def primal_relative_kkt_residual(X, B, mu, Z, eps, C, **kwargs):
    U = calc_U(X, B, mu, Z, eps, C)
    return primal_kkt_residual(X, B, mu, Z, eps, C) / (1 + np.linalg.norm(U))

def dual_kkt_residual(Z, C, **kwargs):
    return np.sum(np.clip(np.abs(Z) - C, 0, np.inf))

def dual_relative_kkt_residual(Z, C, A, **kwargs):
    normA = np.linalg.norm(A)
    return dual_kkt_residual(Z, C) / (1 + normA)

def kkt_gap(X, B, Z, A, eps, C, mu, **kwargs):
    U = calc_U(X, B, mu, Z, eps, C)
    return (np.linalg.norm(Z @ B.T + X - A) + np.linalg.norm(prox(U + Z, eps, C, 1) - Z))

def kkt_relative_gap(X, B, Z, A, eps, C, mu, **kwargs):
    U = calc_U(X, B, mu, Z, eps, C)
    normA = np.linalg.norm(A)
    normU = np.linalg.norm(U)
    return kkt_gap(X, B, Z, A, eps, C, mu) / (1 + normA + normU)
