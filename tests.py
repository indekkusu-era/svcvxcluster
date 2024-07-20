import matplotlib.pyplot as plt
from svcvxcluster.criterions import relative_duality_gap, primal_relative_kkt_residual
from svcvxcluster import SvCvxCluster
from svcvxcluster.svcvxcluster import SolverConfig
from sklearn.datasets import make_moons

if __name__ == "__main__":
    X, y = make_moons(10000, noise=0.05)
    clust = SvCvxCluster(20, alpha=0.3,
                        alpha_prime=2, pairing_strat='heuristic', 
                        solver_config=SolverConfig(criterions=[relative_duality_gap, primal_relative_kkt_residual]))
    clust.fit(X)
    plt.scatter(*X.T, c=clust.labels())
    plt.show()
