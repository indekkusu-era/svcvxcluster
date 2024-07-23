import numpy as np
import networkx as nx
from tqdm import tqdm
from .ssnal_utils import ssnal_grad, prox, dprox
from .ssnal_algorithms import armijo_line_search, armijo_dual, ssnal_cg
# The below code is used in experiments on parallel computing
# from ...experimentals.lab import ssnal_cg
from ...criterions import evaluate_criterions, primal_relative_kkt_residual, dual_relative_kkt_residual, kkt_relative_gap

def sv_cvxcluster_ssnal(A: np.ndarray, eps: float, C: float, graph: nx.Graph, X0=None, Z0=None,
                         mu=1, gamma=0.75, tol=1e-6, criterions=None,
                         armijo_alpha=1, armijo_sigma=0.25, armijo_beta=0.75, armijo_iter=10, 
                         mu_update_tol=1, mu_update_tol_decay=0.95,
                         max_iter=1000, mu_min=1e-5, mu_max=1e5, cgtol_tau=0.618, cgtol_default=1e-5, 
                         parallel=False, verbose=True):
    if criterions is None:
        criterions = [primal_relative_kkt_residual, dual_relative_kkt_residual, kkt_relative_gap]
    incidence_matrix = nx.incidence_matrix(graph, oriented=True)
    if X0 is None:
        X = A.copy()
    else:
        X = X0.copy()
    if Z0 is None:
        Z = np.zeros((A.shape[0], incidence_matrix.shape[1]))
    else:
        Z = Z0.copy()
    dX = None
    n = A.shape[1]; d = A.shape[0]
    j = 0
    normA = np.linalg.norm(A)
    for i in (pbar := (tqdm(range(max_iter)) if verbose else range(max_iter))):
        BX = X @ incidence_matrix
        prox_var = BX / mu + Z
        dual_prox = prox(prox_var, eps, C, 1 / mu)
        U = mu * (prox_var - dual_prox)
        BXDiff = BX - U
        gradX = ssnal_grad(X, A, incidence_matrix, dual_prox)
        Q = dprox(prox_var, eps, C, 1 / mu)
        normgrad = np.linalg.norm(gradX)
        normgradZ = np.linalg.norm(BXDiff)
        cg_tol = min(cgtol_default, normgrad ** (1 + cgtol_tau))
        dX = ssnal_cg(incidence_matrix, gradX, mu, n, Q, dX, cg_tol, parallel=parallel, d=d)
        dZ = (BXDiff + Q * (dX @ incidence_matrix)) / mu
        alpha = armijo_line_search(X, A, Z, incidence_matrix, eps, C, mu, gradX, dX,
                                alpha0=armijo_alpha, beta=armijo_beta, sigma=armijo_sigma, max_iter=armijo_iter)
        beta = armijo_dual(X, A, Z, incidence_matrix, eps, C, mu, BXDiff, dZ,
                                alpha0=armijo_alpha, beta=armijo_beta, sigma=armijo_sigma, max_iter=armijo_iter)
        X += dX * alpha
        Z += dZ * beta
        # Update Optimality Condition
        crit = max(evaluate_criterions(X, incidence_matrix, Z, A, eps, C, mu, U=U, 
                                       BXDiff=BXDiff, normA=normA, normgradZ=normgradZ, 
                                       normU=np.linalg.norm(U), list_criterions=criterions))
        if verbose:
            pbar.set_description(f"mu: {mu:.6f} | criterion: {crit:.6f}")
        # Check Optimality Condition
        if crit < tol:
            break
        if max(normgrad, normgradZ) < mu_update_tol * (mu_update_tol_decay ** j) * min(1, np.sqrt(mu)):
            if normgrad < normgradZ:
                mu *= gamma
                mu = max(mu_min, mu)
            else:
                mu /= gamma
                mu = min(mu_max, mu)
            j += 1
    return X, Z

