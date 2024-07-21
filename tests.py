from sklearn.datasets import make_moons
from svcvxcluster import SvCvxCluster

if __name__ == "__main__":
    X, y = make_moons(200000, noise=0.05)
    clust = SvCvxCluster(10, alpha=0.4, alpha_prime=1)
    clust.fit(X, warm_start=False)
    print((clust.labels() == y).mean())
