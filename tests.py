import numpy as np
from svcvxcluster import SvCvxCluster
from svcvxcluster.svcvxcluster import SolverConfig
from svcvxcluster.benchmark.spherical_shell import generate_semisphere
from sklearn.metrics import adjusted_rand_score
from time import perf_counter_ns
import matplotlib.pyplot as plt

np.random.seed(1234)

if __name__ == "__main__":
    X, y = generate_semisphere(1.0, 1.4, 1.6, 2.0, 100)
    X = X.T
    clust = SvCvxCluster(10, eps=0.05, C=0.25, 
                         warm_start_solver_config=SolverConfig(gamma=0.95, tol=0.1, max_iter=25), 
                         solver_config=SolverConfig(gamma=0.5, mu_update_tol=1, 
                                        armijo_sigma=0.5, mu_update_tol_decay=0.5, armijo_iter=10, parallel=True))
    t = perf_counter_ns()
    clust.fit(X)
    print((perf_counter_ns() - t) / 1e9)
    print(adjusted_rand_score(y, clust.labels_))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*X.T, c=clust.labels_)
    plt.show()
