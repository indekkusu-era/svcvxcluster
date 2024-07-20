import numpy as np
from svcvxcluster.benchmark.halfmoon import benchmark_moons

if __name__ == "__main__":
    times = benchmark_moons(5000, [0], np.arange(0.2, 2.2, 0.2))
    print(times.groupby('eps').mean())
