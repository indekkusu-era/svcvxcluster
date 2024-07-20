"""
Replication of the Methodology from Convex Clustering [1]

[1] D.F. Sun, K.C. Toh, and Y.C. Yuan, Convex clustering: model, theoretical guarantee and efficient algorithm, Journal of Machine Learning Research, 22(9):1?32, 2021.
"""

import pandas as pd
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from time import perf_counter_ns
from ..svcvxcluster import SvCvxCluster


def benchmark_moons(n_samples, nscale_eps, nscale_C):
    X, y = make_moons(n_samples, noise=0.05, shuffle=True)
    model = SvCvxCluster(10, alpha=0.4, alpha_prime=1, pairing_strat='heuristic')
    t = perf_counter_ns()
    model.fit(X) # fit the default model
    nn_graph = model._graph
    solve_time = []
    xbar = model.Xbar_
    Z = model.Z_
    for scale1 in nscale_eps:
        for scale2 in nscale_C:
            model2 = SvCvxCluster(nn='precomputed', eps=scale1, C=scale2)
            t = perf_counter_ns()
            model2.fit(X, graph=nn_graph, warm_start=False, X0=xbar, Z0=Z)
            t2 = (perf_counter_ns() - t) / 1e9
            xbar = model2.Xbar_; Z = model2.Z_
            rand_score = adjusted_rand_score(y, model2.labels())
            solve_time.append({'eps': scale1, 'C': scale2, 'solve_time': t2, 'rand_score': rand_score})
    return pd.DataFrame(solve_time)


__all__ = ['benchmark_moons']
