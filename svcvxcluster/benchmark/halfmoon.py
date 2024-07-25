"""
Replication of the Methodology from Convex Clustering [1]

[1] D.F. Sun, K.C. Toh, and Y.C. Yuan, Convex clustering: model, theoretical guarantee and efficient algorithm, Journal of Machine Learning Research, 22(9):1?32, 2021.
"""
from sklearn.datasets import make_moons

def load_moons(n_samples):
    return make_moons(n_samples, noise=0.05)
