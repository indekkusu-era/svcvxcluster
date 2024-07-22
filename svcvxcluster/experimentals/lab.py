# this is for experimental space for 
# algorithms, etc.

# NOTE: This is currently planning for parallel implementation

import numpy as np
import numba as nb
from scipy.sparse import identity as spI
from scipy.sparse import linalg as splinalg

# TODO: implement shared memory
@nb.njit()
def i_mubpbx(rowb, idxptrb, datab, nrow, idxp, datap, x, mu):
    res = x
    for row in range(nrow-1):
        for row2 in range(row, nrow-1):
            rang1 = np.arange(rowb[row], rowb[row+1])
            rang2 = np.arange(rowb[row2], rowb[row2+1])
            ptr1 = idxptrb[rang1]
            ptr2 = idxptrb[rang2]
            data1 = datab[rang1]
            data2 = datab[rang2]
            idx = np.intersect1d(ptr1, ptr2)
            idx = np.intersect1d(idx, idxp)
            result = mu * np.sum(data1[idx] * data2[idx] * datap[idx]) * x[row2]
            res[row] += result
    return res

# @nb.njit
def ssnal_cg_parallel(B, grad, mu, n, subgrad_P, X0=None, tol=1e-5):
    B = B.T
    for i, row in enumerate(subgrad_P):
        idk = (i_mubpbx(B.indptr, B.indices, B.data, n, row.indices, row.data, X0[i], 1 / mu))
        print(type(idk))

def ssnal_cg(B, grad, mu, n, subgrad_P, X0=None, tol=1e-5, parallel=False):
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
    else:
        return ssnal_cg_parallel(B, grad, mu, n, subgrad_P, dX, tol)

    return dX
