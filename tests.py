from svcvxcluster import SvCvxCluster
from svcvxcluster.solvers import SSNAL
from svcvxcluster.svcvxcluster import SolverConfig
from svcvxcluster.benchmark.halfmoon import load_moons
from svcvxcluster.benchmark.spherical_shell import load_semisphere
from svcvxcluster.benchmark.large_dataset import load_large
from svcvxcluster.experimentals.lab import thread_sv_cvx_cluster_saddle
from svcvxcluster.solvers import Thread_SSNAL
from sklearn.metrics import adjusted_rand_score
from time import perf_counter_ns
import matplotlib.pyplot as plt

import numpy as np
np.random.seed(727)

if __name__ == "__main__":
    X, y = load_semisphere(1.0, 1.4, 1.6, 2.0, 2000)
    clust = SvCvxCluster(10, alpha=0.4, alpha_prime=1, solver=thread_sv_cvx_cluster_saddle)
    t = perf_counter_ns()
    clust.fit(X, warm_start=False)
    print("Solve Time:", (perf_counter_ns() - t) / 1e9)
    print("Adjusted Rand Score:", adjusted_rand_score(y, clust.labels_))
    # plt.scatter(*X.T, c=clust.labels_)
    # plt.show()

