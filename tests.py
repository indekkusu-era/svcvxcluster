import numpy as np
import matplotlib.pyplot as plt
from svcvxcluster.criterions import relative_duality_gap, primal_relative_kkt_residual
from svcvxcluster import SvCvxCluster
from svcvxcluster.svcvxcluster import SolverConfig
from svcvxcluster.benchmark.halfmoon import benchmark_moons

if __name__ == "__main__":
    times = benchmark_moons(1000, np.logspace(-3, 1, 10), np.arange(0.2, 2.2, 0.2))
    print(times.groupby('eps').mean())
