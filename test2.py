import numpy as np
from scipy import sparse
from numba import njit

@njit
def print_csr(A, iA, jA):
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            print(row, jA[i], A[i])

@njit
def i_mubpbx(rowb, idxptrb, datab, nrow, idxp, datap, x, mu):
    idx_bpbx_result = []
    data_bpbx_result = []
    for row in range(nrow-1):
        for row2 in range(row, nrow-1):
            rang1 = np.arange(rowb[row], rowb[row+1])
            rang2 = np.arange(rowb[row2], rowb[row2+1])
            ptr1 = idxptrb[rang1]
            ptr2 = idxptrb[rang2]
            data1 = datab[rang1]
            data2 = datab[rang2]
            idx = np.intersect1d(ptr1, ptr2, idxp)
            result = mu * np.sum(data1[idx] * data2[idx] * datap) * x + x
            data_bpbx_result.append(result)
            idx_bpbx_result.append(row)
    return idx_bpbx_result, data_bpbx_result

