
import numpy as np
import networkx as nx
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from scipy.sparse import linalg as splinalg
from scipy.sparse import identity as spI
from ilupp import IChol0Preconditioner
from ..ssnal.ssnal_algorithms import armijo_line_search, armijo_dual
from ..ssnal.ssnal_utils import prox, dprox, ssnal_grad
from ...criterions import evaluate_criterions
from ...criterions import primal_relative_kkt_residual, dual_relative_kkt_residual, kkt_relative_gap

def sv_cvxcluster_iter_1d(Ai, eps, C, incidence_matrix, Xi, Zi, mu, I, dX_cg, 
                          cgtol_default, cgtol_tau, tol, 
                          armijo_alpha, armijo_beta, armijo_sigma, armijo_iter):
    X = Xi.copy()
    for i in range(10):
        BX = X @ incidence_matrix
        prox_var = BX / mu + Zi
        dual_prox = prox(prox_var, eps, C, 1 / mu)
        U = mu * (prox_var - dual_prox)
        gradX = ssnal_grad(X, Ai, incidence_matrix, dual_prox)
        Q = dprox(prox_var, eps, C, 1 / mu)
        # CG
        BP = incidence_matrix * Q
        sub_laplacian = BP @ BP.T
        mat = I + sub_laplacian / mu
        normgrad = np.linalg.norm(gradX)
        if normgrad < tol:
            break
        cg_tol = min(cgtol_default, normgrad ** (1 + cgtol_tau))
        dX, exit_code = splinalg.cg(mat, -gradX, atol=cg_tol, x0=dX_cg)
        alpha = armijo_line_search(X, Ai, Zi, incidence_matrix, eps, C, mu, gradX, dX,
                                    alpha0=armijo_alpha, beta=armijo_beta, sigma=armijo_sigma, max_iter=armijo_iter)
        X += alpha * dX
    BX = X @ incidence_matrix
    prox_var = BX / mu + Zi
    dual_prox = prox(prox_var, eps, C, 1 / mu)
    U = mu * (prox_var - dual_prox)
    BXDiff = BX - U
    return X, Zi + BXDiff / mu


def thread_sv_cvx_cluster(A: np.ndarray, eps: float, C: float, graph: nx.Graph, X0=None, Z0=None,
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
        chol_prec = IChol0Preconditioner(spI(n) + incidence_matrix @ incidence_matrix.T / mu)
        X0 = -ssnal_grad(A, A, incidence_matrix, prox(A @ incidence_matrix / mu, eps, C, 1/mu))
        for i in range(d):
            u = X0[i].copy()
            chol_prec.apply(u)
            X0[i] = u
    if Z0 is None:
        Z0 = np.zeros((d, incidence_matrix.shape[1]))
    X = X0.copy()
    dX = np.zeros(X.shape)
    Z = Z0.copy()
    I = spI(n)
    normA = np.linalg.norm(A)
    primal_update = 0
    dual_update = 0
    primal_residual_sequence = []
    dual_residual_sequence = []
    duality_gap_sequence = [] 
    for iteration in (pbar := (tqdm(range(max_iter)) if verbose else range(max_iter))):
        threads = []
        with ThreadPoolExecutor(max_workers=None) as executor:
            for i in range(d):
                threads.append(executor.submit(sv_cvxcluster_iter_1d, 
                                            A[i], eps, C, incidence_matrix, 
                                            X[i], Z[i], mu, I, dX[i], 
                                            cgtol_default, cgtol_tau, mu_update_tol * (mu_update_tol_decay ** iteration) * min(1, np.sqrt(mu)), armijo_alpha, 
                                            armijo_beta, armijo_sigma, armijo_iter)
                                        )
            
            for i, task in enumerate(threads):
                X[i], Z[i] = task.result()
        BX = X @ incidence_matrix
        prox_var = BX / mu + Z
        dual_prox = prox(prox_var, eps, C, 1 / mu)
        U = mu * (prox_var - dual_prox)
        BXDiff = BX - U
        normgradZ = np.linalg.norm(BXDiff)
        gradX = ssnal_grad(X, A, incidence_matrix, dual_prox)
        normgrad = np.linalg.norm(gradX)
        primal_feas, dual_feas, ngap = evaluate_criterions(X, incidence_matrix, Z, A, eps, C, mu, U=U, 
                                        BXDiff=BXDiff, normA=normA, normgradZ=normgradZ, 
                                        normU=np.linalg.norm(U), list_criterions=criterions)
        crit = max(primal_feas, dual_feas, ngap)
        primal_residual_sequence.append(primal_feas)
        dual_residual_sequence.append(dual_feas)
        duality_gap_sequence.append(ngap)
        if verbose:
            pbar.set_description(f"mu: {mu:.6f} | criterion: {crit:.6f}")
        # Check Optimality Condition
        if crit < tol:
            break
        update = int(primal_relative_kkt_residual(X, incidence_matrix, mu, Z, eps, C, U, BXDiff) < dual_relative_kkt_residual(Z, C, A))
        primal_update += update
        dual_update += (1 - update)
        if iteration % 5 == 4:
            if primal_update > dual_update * 1.2:
                mu /= gamma
                mu = min(mu_max, mu)
                primal_update = 0
            elif dual_update > primal_update * 1.2:
                mu *= gamma
                mu = max(mu_min, mu)
                dual_update = 0
    return X, Z, (primal_residual_sequence, dual_residual_sequence, duality_gap_sequence)


