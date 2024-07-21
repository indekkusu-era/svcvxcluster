import numpy as np
from ..utils.objective_fn import obj_primal, obj_dual

def duality_gap(X, B, Z, A, eps, C, mu, **kwargs):
    return np.abs(obj_primal(X, B, Z, A, eps, C, mu, **kwargs) - obj_dual(X, B, Z, A, eps, C, mu, **kwargs))

def relative_duality_gap(X, B, Z, A, eps, C, mu, **kwargs):
    objprimal = obj_primal(X, B, Z, A, eps, C, mu)
    objdual = obj_dual(X, B, Z, A, eps, C, mu)
    return np.abs(objprimal - objdual) / (1 + np.abs(objprimal) + np.abs(objdual)) 
