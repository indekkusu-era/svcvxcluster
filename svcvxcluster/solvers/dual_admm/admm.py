import numpy as np
import networkx as nx
from tqdm import tqdm
from .admm_algorithms import cg
from .admm_utils import prox, Tstar, T
from ...criterions import relative_duality_gap
from ...criterions import evaluate_criterions

def initialize(A, G, p0=None, l0=None, tol=None):
    if tol is None:
        tol = 0.05
    incidence_matrix = nx.incidence_matrix(G, oriented=True)
    laplacian_matrix = (incidence_matrix @ incidence_matrix.T)
    m = incidence_matrix.shape[1]
    d = A.shape[0]
    if p0 is None:
        P = np.zeros((d, 2*m))
    else:
        P = p0.copy()
    if l0 is None:
        L = np.zeros((d, 2*m))
    else:
        L = l0.copy()
    tau = (1 + np.sqrt(5)) / 2
    TTA = 2 * A @ laplacian_matrix
    normsqA = np.linalg.norm(A) ** 2
    return incidence_matrix, laplacian_matrix, TTA, tau, normsqA, G, P, L, tol, m

def sv_cvxcluster_admm(A: np.ndarray, eps: float, C: float, graph: nx.Graph, X0=None, Z0=None,
                         mu=1, gamma=0.95, tol=1e-4, max_iter=1000, criterions=None,
                         armijo_alpha=1, armijo_sigma=0.1, armijo_beta=0.75, armijo_iter=10, 
                         mu_update_tol=1e-2, mu_update_tol_decay=0.9,
                         mu_min=1e-3, mu_max=100, cgtol_tau=0.618, cgtol_default=1e-5, 
                         parallel=False, verbose=True):
    if criterions is None:
        criterions = [relative_duality_gap]
    norm_grad = 1
    incidence_matrix, laplacian_matrix, TTA, tau, normsqA, G, P, L, tol, m = initialize(A, graph)
    cg_z = None
    for i in (pbar := tqdm(range(max_iter))):
        Lprox = L - prox(P + mu * L, mu, eps, C) / mu
        tl = T(Lprox, incidence_matrix)
        Z = -TTA + tl
        cg_z, cg_iter = cg(Z.T, laplacian_matrix, incidence_matrix, mu, X0=cg_z, tol=min(1e-5, norm_grad ** 1.618), n_iter=100)
        dP = - P - mu * Lprox + mu * Tstar(A + mu * cg_z.T, incidence_matrix)
        norm_grad = np.linalg.norm(dP)
        P += dP
        L += tau * (P - prox(P + mu * L, mu, eps, C)) / mu
        mu *= gamma
        mu = max(mu_min, mu)
        X = A - T(P, incidence_matrix)
        Z = P[:, :m] - P[:, m:]
        crit = max(evaluate_criterions(X, incidence_matrix, Z, A, eps, C, mu, criterions))
        pbar.set_description(f"cg_iter: {cg_iter} | criterion: {crit:6f}")
        if crit < tol:
            break
    return X, Z