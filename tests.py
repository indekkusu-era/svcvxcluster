import numpy as np
from svcvxcluster import SvCvxCluster
from svcvxcluster.svcvxcluster import SolverConfig
from svcvxcluster.benchmark.large_dataset import load_large
from svcvxcluster.benchmark.spherical_shell import load_semisphere
from svcvxcluster.benchmark.halfmoon import load_moons
from svcvxcluster.experimentals.lab import thread_sv_cvx_cluster
from sklearn.metrics import adjusted_rand_score
from time import perf_counter_ns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

np.random.seed(1234)

if __name__ == "__main__":
    X, y = load_moons(10000) # load_semisphere(1.0, 1.4, 1.6, 2.0, 1000) # load_large(500000, 10)
    # X = StandardScaler().fit_transform(X)
    clust = SvCvxCluster(10, alpha=0.2, alpha_prime=1, 
                         warm_start_solver_config=SolverConfig(gamma=1, tol=0.25, max_iter=100), 
                         solver_config=SolverConfig(gamma=0.75, mu_update_tol=1, 
                                            armijo_sigma=0.25, mu_update_tol_decay=0.75, armijo_iter=10, 
                                            cgtol_tau=1/2, cgtol_default=1e-5, preconditioner='auto', parallel=True
                                        ),
                        # solver=thread_sv_cvx_cluster
                        )
    t = perf_counter_ns()
    clust.fit(X, warm_start=False)
    print("SSNAL:", (perf_counter_ns() - t) / 1e9)
    print(adjusted_rand_score(y, clust.labels_))
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(*X.T, c=clust.labels_)
    # plt.show()
