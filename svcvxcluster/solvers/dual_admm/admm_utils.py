import numpy as np

def T(A, incidence_mat):
    m = incidence_mat.shape[1]
    return (A[:, :m] - A[:, m:]) @ incidence_mat.T

def Tstar(A, incidence_mat):
    AJ = A @ incidence_mat
    return np.hstack([AJ, -AJ])

def prox(P, mu, eps, C):
    return np.clip(P - eps * mu, 0, C)

def dprox(P, mu, eps, C):
    Phat = P - mu * eps
    return np.where((Phat <= 0) | (Phat >= C), 1, 0)

def obj_function(P, L, A, mu, eps, C, inc_mat):
    return 0.5 * np.linalg.norm(T(P, inc_mat) - A) ** 2 + 0.5 * np.linalg.norm((phat := P + mu * L) - (proxP := prox(phat, mu, eps, C))) ** 2 / mu + eps * np.sum(proxP)
