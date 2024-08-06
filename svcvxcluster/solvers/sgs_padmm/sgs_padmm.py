import numpy as np
import networkx as nx
from scipy.sparse import tril, diags, identity, csr_matrix
from scipy.sparse.linalg import spsolve_triangular
from tqdm import tqdm
from .sgs_padmm_utils import prox
from ...criterions import relative_duality_gap
from ...criterions import evaluate_criterions

def sv_cvx_cluster_sgs_padmm(A: np.ndarray, eps: float, C: float, graph: nx.Graph, X0=None, Z0=None,
                         mu=1, gamma=0.95, tol=1e-4, max_iter=1000, criterions=None, preconditioner=None,
                         armijo_alpha=1, armijo_sigma=0.1, armijo_beta=0.75, armijo_iter=10, 
                         mu_update_tol=1e-2, mu_update_tol_decay=0.9,
                         mu_min=1e-3, mu_max=100, cgtol_tau=0.618, cgtol_default=1e-5, 
                         parallel=False, verbose=True):
    if criterions is None:
        criterions = [relative_duality_gap]
    incidence_matrix = nx.incidence_matrix(graph)
    laplacian_matrix = incidence_matrix @ incidence_matrix.T
    ltri_laplacian = tril(laplacian_matrix, format='csr')
    I = identity(A.shape[1])
    degree_diag = diags(np.array(graph.degree)[:, 1], format='csr')
    normA = np.linalg.norm(A)
    if X0 is None:
        X = A.copy()
    else:
        X = X0.copy()
    if Z0 is None:
        Z = np.zeros((A.shape[0], incidence_matrix.shape[1]))
    else:
        Z = Z0.copy()
    BX = X @ incidence_matrix
    prox_var = BX / mu + Z
    dual_prox = prox(prox_var, eps, C, 1 / mu)
    U = mu * (prox_var - dual_prox)
    BXDiff = BX - U
    for i in (pbar := (tqdm(range(max_iter)) if verbose else range(max_iter))):
        gradx = X - A + (Z + BXDiff / mu) @ incidence_matrix.T
        # Calculate H inv Grad
        dx = -gradx.T
        M = I + ltri_laplacian / mu
        spsolve_triangular(M, dx, overwrite_b=True)
        dx += degree_diag @ dx / mu
        spsolve_triangular(csr_matrix(M.T), dx, lower=False, overwrite_b=True)
        X += dx.T
        BX = X @ incidence_matrix
        prox_var = BX / mu + Z
        dual_prox = prox(prox_var, eps, C, 1 / mu)
        U = mu * (prox_var - dual_prox)
        BXDiff = BX - U
        Z += BXDiff / mu
        mu *= gamma
        crit = max(evaluate_criterions(X, incidence_matrix, Z, A, eps, C, mu, U=U, 
                                       BXDiff=BXDiff, normA=normA, 
                                       normU=np.linalg.norm(U), list_criterions=criterions))
        if verbose:
            pbar.set_description(f"criterion: {crit:.6f}")
        if crit < tol:
            break
    return X, Z
