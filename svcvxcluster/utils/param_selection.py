from sklearn.metrics import pairwise_distances
import numpy as np

def auto_select_params(A, nn, inc_mat, alpha=0.05, alpha_prime=1):
    pw = np.abs(A.T @ inc_mat).max(axis=0)
    diameter = np.max(A.max(axis=1) - A.min(axis=1))
    max_nn = pw.max()
    return (eps := max_nn * alpha), (diameter - eps) / (2 * nn) * alpha_prime
