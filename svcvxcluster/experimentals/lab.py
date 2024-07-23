# this is for experimental space for 
# algorithms, etc.

# NOTE: This is currently planning for parallel implementation

import numpy as np
import numba as nb
from scipy.sparse import identity as spI
from scipy.sparse import linalg as splinalg
from scipy.sparse import csr_matrix

@nb.njit(parallel=True)
def i_mubpbx(rowb, idxptrb, datab, nrow, idxp, datap, x, mu):
    xprime = x.copy()
    for row in nb.prange(nrow-1):
        for row2 in nb.prange(nrow-1):
            rang1 = np.arange(rowb[row], rowb[row+1])
            rang2 = np.arange(rowb[row2], rowb[row2+1])
            ptr1 = idxptrb[rang1]
            ptr2 = idxptrb[rang2]
            data1 = datab[rang1]
            data2 = datab[rang2]
            i = 0; j = 0; k = 0
            maxptr = max(len(idxp), len(ptr1), len(ptr2))
            for _ in range(maxptr):
                if i >= len(ptr1) or j >= len(ptr2) or k >= len(idxp):
                    break
                curidx1, curidx2, curidxp = ptr1[i], ptr2[j], idxp[k]
                minidx = min(curidx1, curidx2, curidxp)
                if (curidx1 == curidx2) and (curidx2 == curidxp):
                    xprime[row] += mu * data1[i] * data2[j] * datap[k] * x[row2]
                    i += 1; j += 1; k += 1
                    continue
                if curidx1 == minidx:
                    i += 1
                if curidx2 == minidx:
                    j += 1
                if curidxp == minidx:
                    k += 1
    return xprime

@nb.njit()
def sparse_cg_iter(b_indptr, b_idx, b_data, p_idx, p_data, X, R, P, n, mu):
    Ap = i_mubpbx(b_indptr, b_idx, b_data, n, p_idx, p_data, P, mu)
    alpha = np.sum(R * R) / np.sum(P * Ap)
    X1 = X + alpha * P
    R1 = R - alpha * Ap
    beta = np.sum(R1 ** 2) / np.sum(R ** 2)
    P1 = R1 + beta * P
    return X1, R1, P1

@nb.njit(parallel=True)
def sparse_cg(b_indptr, b_idx, b_data, p_indptr, p_idx, p_data, grad, dX, mu, n, d, niter, tol):
    for i in nb.prange(d-1):
        startP = p_indptr[i]
        endP = p_indptr[i+1]
        rangP = np.arange(startP, endP)
        idxP = p_idx[rangP]
        dataP = p_data[rangP]
        b_target = -grad[i]
        X = dX[i].copy()
        R = b_target - i_mubpbx(b_indptr, b_idx, b_data, n, idxP, dataP, X, mu)
        P = R.copy()
        for j in range(niter):
            if np.linalg.norm(R) < tol:
                break
            X, R, P = sparse_cg_iter(b_indptr, b_idx, b_data, idxP, dataP, X, R, P, n, mu)
        dX[i] = X
    return dX

def ssnal_cg_parallel(B, grad, mu, n, d, subgrad_P, X0, tol=1e-5, max_iter=100):
    return sparse_cg(B.indptr, B.indices, B.data, 
                     subgrad_P.indptr, subgrad_P.indices, subgrad_P.data, 
                     grad, X0, 1 / mu, n, d, max_iter, tol)

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
        return dX
    else:
        B = csr_matrix(B.T) # TODO: Move this to real code
        new_dx = ssnal_cg_parallel(B, grad, mu, n, d, subgrad_P, dX, tol)
        return new_dx

