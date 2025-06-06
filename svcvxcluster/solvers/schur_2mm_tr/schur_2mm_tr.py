import threading
from concurrent.futures import ThreadPoolExecutor
import warnings

import numpy as np
import networkx as nx
from tqdm import tqdm
import scipy.sparse.linalg as splinalg
from scipy.sparse import identity as spI
from scipy.sparse import csr_array
from ..ssnal.ssnal_utils import ssnal_grad, prox, dprox
from ..ssnal.ssnal_algorithms import armijo_line_search, armijo_dual, obj_function
from ...criterions import evaluate_criterions, primal_relative_kkt_residual, dual_relative_kkt_residual, kkt_relative_gap
from ...criterions.gap import obj_dual

def make_jinv(q: csr_array, tau: float, mu: float):
    j = np.ones(q.shape) / ((1 + tau) * mu)
    j[q.nonzero()] = 1 / (tau * mu)
    return j

def calculate_trust_region(X, B, Z, A, eps, C, mu, dZ, dX):
    fx = -obj_dual(X, B, Z, A, eps, C, mu)
    fxp = -obj_dual(X, B, Z + dZ, A, eps, C, mu)
    mp = -obj_function(X + dX, A, Z + dZ, B, eps, C, mu)
    return (fx - fxp) / (fx - mp)

def sv_cvxcluster_ssnal2_iter_1d(Ai, eps, C, incidence_matrix, Xi, Zi, mu, tau, I, dX_cg, 
                          cgtol_default, cgtol_tau, 
                          armijo_alpha, armijo_beta, armijo_sigma, armijo_iter):
    BX = Xi @ incidence_matrix
    prox_var = BX / mu + Zi
    dual_prox = prox(prox_var, eps, C, 1 / mu)
    U = mu * (prox_var - dual_prox)
    BXDiff = BX - U
    gradX = ssnal_grad(Xi, Ai, incidence_matrix, dual_prox)
    Q = dprox(prox_var, eps, C, 1 / mu)
    # CG
    BP = incidence_matrix.multiply(Q)
    jinv = make_jinv(Q, tau, mu)
    mat = I + (BP @ BP.T) / mu * (1 + 1 / tau) # + PBJ @ BP.T
    normgrad = np.linalg.norm(gradX)
    cg_tol = min(cgtol_default, normgrad ** (1 + cgtol_tau))
    inv_diag = 1 / mat.diagonal()
    one_onet = 1 / mat.sum()
    prec = splinalg.LinearOperator(dtype=np.float64, shape=mat.shape, matvec=lambda x: x * inv_diag + np.sum(x) * one_onet)
    dX, exit_code = splinalg.cg(mat, - gradX - (BXDiff @ BP.T) / (tau * mu), atol=cg_tol, x0=dX_cg, M=prec)
    bdx = dX @ incidence_matrix
    dZ = jinv * (bdx * Q + BXDiff)
    alpha = armijo_line_search(Xi, Ai, Zi, incidence_matrix, eps, C, mu, gradX, dX,
                                alpha0=armijo_alpha, beta=armijo_beta, sigma=armijo_sigma, max_iter=armijo_iter)
    beta = armijo_dual(Xi, Ai, Zi, incidence_matrix, eps, C, mu, BXDiff, dZ,
                                alpha0=armijo_alpha, beta=armijo_beta, sigma=armijo_sigma, max_iter=armijo_iter)
    return dX * alpha, dZ * beta, dX

def thread_schur_2mm_tr(A: np.ndarray, eps: float, C: float, graph: nx.Graph, X0=None, Z0=None,
                         mu=1, gamma=0.75, tol=1e-6, criterions=None, preconditioner='auto',
                         armijo_alpha=1, armijo_sigma=0.25, armijo_beta=0.75, armijo_iter=10, 
                         mu_update_tol=1, mu_update_tol_decay=0.95,
                         max_iter=1000, mu_min=1e-5, mu_max=1e5, cgtol_tau=0.618, cgtol_default=1e-5, 
                         parallel=True, verbose=True):
    if not parallel or preconditioner != 'auto':
        warnings.warn("For Threading SVCvxCluster, the 'parallel' parameter will automatically be set to True and 'preconditioner' will automatically be set to 'auto'", UserWarning)
    d = A.shape[0]; n = A.shape[1]
    if criterions is None:
        criterions = [primal_relative_kkt_residual, dual_relative_kkt_residual, kkt_relative_gap]
    incidence_matrix = nx.incidence_matrix(graph, oriented=True)
    if X0 is None:
        X0 = A.copy()
    if Z0 is None:
        Z0 = np.zeros((d, incidence_matrix.shape[1]))
    X = X0.copy()
    dX = np.zeros(X.shape)
    Z = Z0.copy()
    I = spI(n)
    normA = np.linalg.norm(A)
    dxx = np.zeros_like(X); dzz = np.zeros_like(Z)
    tau = 1 / 9
    primal_residual_sequence = []
    dual_residual_sequence = []
    duality_gap_sequence = [] 
    for iterations in (pbar := (tqdm(range(max_iter)) if verbose else range(max_iter))):
        threads = []
        with ThreadPoolExecutor(max_workers=None) as executor:
            for i in range(d):
                threads.append(executor.submit(sv_cvxcluster_ssnal2_iter_1d, 
                                            A[i], eps, C, incidence_matrix,
                                            X[i], Z[i], mu, tau, I, dX[i], 
                                            cgtol_default, cgtol_tau, armijo_alpha, 
                                            armijo_beta, armijo_sigma, armijo_iter)
                                        )
            
            for i, task in enumerate(threads):
                dxx[i], dzz[i], dX[i] = task.result()
        rho = calculate_trust_region(X, incidence_matrix, Z, A, eps, C, mu, dzz, dxx)
        X += dxx
        Z += dzz
        BX = X @ incidence_matrix
        prox_var = BX / mu + Z
        dual_prox = prox(prox_var, eps, C, 1 / mu)
        U = mu * (prox_var - dual_prox)
        BXDiff = BX - U
        normgradZ = np.linalg.norm(BXDiff)
        gradX = ssnal_grad(X, A, incidence_matrix, dual_prox)
        primal_feas, dual_feas, ngap = evaluate_criterions(X, incidence_matrix, Z, A, eps, C, mu, U=U, 
                                        BXDiff=BXDiff, normA=normA, normgradZ=normgradZ, 
                                        normU=np.linalg.norm(U), list_criterions=criterions)
        crit = max(primal_feas, dual_feas, ngap)
        primal_residual_sequence.append(primal_feas)
        dual_residual_sequence.append(dual_feas)
        duality_gap_sequence.append(ngap)
        if verbose:
            pbar.set_description(f"mu: {mu:.6f} | criterion: {crit:.6f} | rho: {rho:.6f} | tau: {tau:.6f}")
        # Check Optimality Condition
        if crit < tol:
            break
        if rho > 0.75:
            mu *= gamma
            mu = max(mu_min, mu)
        elif rho < 0.25:
            mu /= gamma
            mu = min(mu_max, mu)
    return X, Z, (primal_residual_sequence, dual_residual_sequence, duality_gap_sequence)
