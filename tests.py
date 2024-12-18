from svcvxcluster import SvCvxCluster
from svcvxcluster.solvers import SSNAL
from svcvxcluster.svcvxcluster import SolverConfig
from svcvxcluster.benchmark.halfmoon import load_moons
from svcvxcluster.benchmark.spherical_shell import load_semisphere
from svcvxcluster.benchmark.large_dataset import load_large
from svcvxcluster.solvers import Thread_SSNAL, thread_schur_2mm_tr
from sklearn.metrics import adjusted_rand_score
from time import perf_counter_ns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
np.random.seed(727)

if __name__ == "__main__":
    samples = []
    for nsamples in [50000, 200000, 500000]:
        X, y = load_moons(nsamples)
        clust = SvCvxCluster(20, alpha=0.7, alpha_prime=10, solver=Thread_SSNAL)
        t = perf_counter_ns()
        clust.fit(X, warm_start=False)
        print(f"eps={clust._eps}, C={clust._C}")
        print("Solve Time:", (ssnal_runtime := (perf_counter_ns() - t) / 1e9))
        print("Adjusted Rand Score:", adjusted_rand_score(y, clust.labels_))
        primal_seq, dual_seq, gap_seq = clust.relative_residual

        clust = SvCvxCluster(20, alpha=0.7, alpha_prime=10, solver=thread_schur_2mm_tr)
        t = perf_counter_ns()
        clust.fit(X, warm_start=False)
        print("Solve Time:", (schur_ssnal_tr := (perf_counter_ns() - t) / 1e9))
        print("Adjusted Rand Score:", adjusted_rand_score(y, clust.labels_))
        primal_seq2, dual_seq2, gap_seq2 = clust.relative_residual

        samples.append({'dataset_id': 'moon', 
                        'n_samples': nsamples, 
                        'n_features': 2, 
                        'ssnal_runtime': ssnal_runtime, 
                        'schur_ssnal_tr_runtime': schur_ssnal_tr})
    
    for nsamples in [25000, 100000, 250000]:
        X, y = load_semisphere(1.0, 1.4, 1.6, 2.0, nsamples)
        clust = SvCvxCluster(10, eps=0, C=50, solver=Thread_SSNAL)
        t = perf_counter_ns()
        clust.fit(X, warm_start=False)
        print(f"eps={clust._eps}, C={clust._C}")
        print("Solve Time:", (ssnal_runtime := (perf_counter_ns() - t) / 1e9))
        print("Adjusted Rand Score:", adjusted_rand_score(y, clust.labels_))
        primal_seq, dual_seq, gap_seq = clust.relative_residual

        clust = SvCvxCluster(10, eps=0, C=50, solver=thread_schur_2mm_tr)
        t = perf_counter_ns()
        clust.fit(X, warm_start=False)
        print("Solve Time:", (schur_ssnal_tr := (perf_counter_ns() - t) / 1e9))
        print("Adjusted Rand Score:", adjusted_rand_score(y, clust.labels_))
        primal_seq2, dual_seq2, gap_seq2 = clust.relative_residual

        plt.figure()
        plt.title("SSNAL vs Schur 2MM TR Relative KKT Residual")
        plt.plot(gap_seq, label='SSNAL')
        plt.plot(gap_seq2, label='Schur 2MM TR')
        plt.yscale('log')
        plt.legend()
        plt.figure()
        plt.title("SSNAL vs Schur 2MM TR Relative Criterion")
        plt.plot(np.max(np.vstack(([primal_seq], [dual_seq], [gap_seq])), axis=0), label='SSNAL')
        plt.plot(np.max(np.vstack(([primal_seq2], [dual_seq2], [gap_seq2])), axis=0), label='Schur 2MM TR')
        plt.yscale('log')
        plt.legend()
    
    plt.show()

    for (nsamples, n_features) in [(50000, 10), (200000, 10), (500000, 10)]:
        X, y = load_large(nsamples, n_features)
        clust = SvCvxCluster(10, alpha=0.7, alpha_prime=10, solver=Thread_SSNAL)
        t = perf_counter_ns()
        clust.fit(X, warm_start=False)
        print(f"eps={clust._eps}, C={clust._C}")
        print("Solve Time:", (ssnal_runtime := (perf_counter_ns() - t) / 1e9))
        print("Adjusted Rand Score:", adjusted_rand_score(y, clust.labels_))
        primal_seq, dual_seq, gap_seq = clust.relative_residual

        clust = SvCvxCluster(10, alpha=0.7, alpha_prime=10, solver=thread_schur_2mm_tr)
        t = perf_counter_ns()
        clust.fit(X, warm_start=False)
        print("Solve Time:", (schur_ssnal_tr := (perf_counter_ns() - t) / 1e9))
        print("Adjusted Rand Score:", adjusted_rand_score(y, clust.labels_))
        primal_seq2, dual_seq2, gap_seq2 = clust.relative_residual

        samples.append({'dataset_id': 'blob', 
                        'n_samples': nsamples, 
                        'n_features': n_features, 
                        'ssnal_runtime': ssnal_runtime, 
                        'schur_ssnal_tr_runtime': schur_ssnal_tr})
    
    pd.DataFrame(samples).to_csv('runtimes.csv', index=False)

