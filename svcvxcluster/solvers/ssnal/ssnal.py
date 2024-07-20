import numpy as np
import networkx as nx
from tqdm import tqdm
from .ssnal_utils import ssnal_grad, prox, dprox
from .ssnal_algorithms import ssnal_cg, armijo_line_search, armijo_dual
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
    n = A.shape[1]
    j = 0
    for i in (pbar := (tqdm(range(max_iter)) if verbose else range(max_iter))):
        BX = X @ incidence_matrix
        prox_var = BX / mu + Z
        gradX = ssnal_grad(X, A, incidence_matrix, prox_var, mu, eps, C)
        gradZ = - mu * (Z - prox(prox_var, eps, C, 1 / mu))
        Q = dprox(prox_var, eps, C, 1 / mu)
        normgrad = np.linalg.norm(gradX)
        cg_tol = min(cgtol_default, normgrad ** (1 + cgtol_tau))
        dX = ssnal_cg(incidence_matrix, gradX, mu, n, Q, dX, cg_tol)
        dZ = (gradZ + Q * (dX @ incidence_matrix)) / mu
        alpha = armijo_line_search(X, A, Z, incidence_matrix, eps, C, mu, gradX, dX,
                                alpha0=armijo_alpha, beta=armijo_beta, sigma=armijo_sigma, max_iter=armijo_iter)
        beta = armijo_dual(X + dX * alpha, A, Z, incidence_matrix, eps, C, mu, gradZ, dZ,
                                alpha0=armijo_alpha, beta=armijo_beta, sigma=armijo_sigma, max_iter=armijo_iter)
        X += dX * alpha
        Z += dZ * beta
        BX = X @ incidence_matrix
        prox_var = BX / mu + Z
        # Update Optimality Condition
        crit = max(evaluate_criterions(X, incidence_matrix, Z, A, eps, C, mu, criterions))
        if verbose:
            pbar.set_description(f"mu: {mu:.6f} | criterion: {crit:.6f}")
        # Check Optimality Condition
        if crit < tol:
            break
        normgradX = np.linalg.norm(gradX)
        normgradZ = np.linalg.norm(gradZ)
        if max(normgradX, normgradZ) < mu_update_tol * (mu_update_tol_decay ** j) * min(1, np.sqrt(mu)):
            if normgradX < normgradZ:
                mu *= gamma
                mu = max(mu_min, mu)
            else:
                mu /= gamma
                mu = min(mu_max, mu)
            j += 1
    return X, Z

