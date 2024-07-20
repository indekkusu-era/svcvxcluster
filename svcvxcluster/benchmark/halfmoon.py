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
    model = SvCvxCluster(30, alpha=0.4, alpha_prime=1)
    t = perf_counter_ns()
    model.fit(X) # fit the default model
    t2 = (perf_counter_ns() - t) / 1e9
    rand_score = adjusted_rand_score(y, model.labels())
    eps, C = model._eps, model._C
    xbar, Z = model.Xbar_, model.Z_
    nn_graph = model._graph
    solve_time = [{'eps': eps, 'C': C, 'solve_time': t2 / 1e9, 'rand_score': rand_score}]
    for scale1 in nscale_eps:
        for scale2 in nscale_C:
            model2 = SvCvxCluster(nn='precomputed', eps=eps*scale1, C=C*scale2)
            t = perf_counter_ns()
            model2.fit(X, graph=nn_graph, warm_start=False, X0=xbar, Z0=Z)
            t2 = (perf_counter_ns() - t) / 1e9
            rand_score = adjusted_rand_score(y, model.labels())
            solve_time.append({'eps': eps*scale1, 'C': C*scale2, 'solve_time': t2, 'rand_score': rand_score})
    return pd.DataFrame(solve_time)


__all__ = ['benchmark_moons']
