"""
Replication of the Methodology from Convex Clustering [1]

[1] D.F. Sun, K.C. Toh, and Y.C. Yuan, Convex clustering: model, theoretical guarantee and efficient algorithm, Journal of Machine Learning Research, 22(9):1?32, 2021.
"""
import numpy as np

def load_semisphere(r1_inner, r2_inner, r1_outer, r2_outer, n):
    # Generate random points for the inner semi-spherical shell
    radius_inner = (r2_inner - r1_inner) * np.random.rand(n) + r1_inner
    theta_inner = 2 * np.pi * np.random.rand(n)
    psi_inner = 0.5 * np.pi * np.random.rand(n)
    
    x_inner = radius_inner * np.sin(psi_inner) * np.cos(theta_inner)
    y_inner = radius_inner * np.sin(psi_inner) * np.sin(theta_inner)
    z_inner = radius_inner * np.cos(psi_inner)

    # Generate random points for the outer semi-spherical shell
    radius_outer = (r2_outer - r1_outer) * np.random.rand(n) + r1_outer
    theta_outer = 2 * np.pi * np.random.rand(n)
    psi_outer = 0.5 * np.pi * np.random.rand(n)
    
    x_outer = radius_outer * np.sin(psi_outer) * np.cos(theta_outer)
    y_outer = radius_outer * np.sin(psi_outer) * np.sin(theta_outer)
    z_outer = radius_outer * np.cos(psi_outer)

    # Combining the data
    A_inner = np.vstack((x_inner, y_inner, z_inner))
    A_outer = np.vstack((x_outer, y_outer, z_outer))
    X = np.hstack((A_inner, A_outer))
    label = np.concatenate((np.ones(n), 2 * np.ones(n)))

    return X.T, label


