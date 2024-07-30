import numpy as np
from svcvxcluster import SvCvxCluster
from svcvxcluster.svcvxcluster import SolverConfig
from svcvxcluster.benchmark.large_dataset import load_large
from svcvxcluster.benchmark.spherical_shell import load_semisphere
from svcvxcluster.benchmark.halfmoon import load_moons
from svcvxcluster.solvers.sgs_padmm import sv_cvx_cluster_sgs_padmm
from svcvxcluster.criterions import primal_relative_kkt_residual, dual_relative_kkt_residual, relative_duality_gap
from sklearn.metrics import adjusted_rand_score
from time import perf_counter_ns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# np.random.seed(1234)

if __name__ == "__main__":
    X, y = load_moons(100000) # load_semisphere(1.0, 1.4, 1.6, 2.0, 20000) # load_large(500000, 10)
    # X = StandardScaler().fit_transform(X)
    clust = SvCvxCluster(10, alpha=0.4, alpha_prime=2, 
                         warm_start_solver_config=SolverConfig(gamma=1, tol=0.05, max_iter=100), 
                         solver_config=SolverConfig(gamma=0.75, mu_update_tol=1, 
                                            armijo_sigma=0.25, mu_update_tol_decay=0.75, armijo_iter=10, 
                                            cgtol_tau=1/2, cgtol_default=1e-5, preconditioner='auto', parallel=True, criterions=[primal_relative_kkt_residual, dual_relative_kkt_residual]
                                        ),
                            solver_warm_start=sv_cvx_cluster_sgs_padmm
                        )
    t = perf_counter_ns()
    clust.fit(X, warm_start=True)
    print("SSNAL:", (perf_counter_ns() - t) / 1e9)
    print(adjusted_rand_score(y, clust.labels_))
    
