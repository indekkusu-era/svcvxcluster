from svcvxcluster import SvCvxCluster
from svcvxcluster.benchmark.halfmoon import load_moons
from sklearn.metrics import adjusted_rand_score
from time import perf_counter_ns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X, y = load_moons(100000)
    clust = SvCvxCluster(10, alpha=0.4, alpha_prime=1)
    t = perf_counter_ns()
    clust.fit(X)
    print("Solve Time:", (perf_counter_ns() - t) / 1e9)
    print("Adjuste Rand Score:", adjusted_rand_score(y, clust.labels_))
    plt.scatter(*X.T, c=clust.labels_)
    plt.show()
    
