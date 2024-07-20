import numpy as np
from svcvxcluster.benchmark.halfmoon import benchmark_moons

if __name__ == "__main__":
    times = benchmark_moons(1000, [0.05], np.arange(0.2, 2.2, 0.2))
    print(times.describe())
