import numpy as np
from tqdm import tqdm
from sklearn.metrics import pairwise_distances_chunked
import networkx as nx

def nn_pairs(A: np.ndarray, nn: int, chunk_size: int = 100):
    def process_chunk(dist_chunk, start):
        indices_chunk = np.argsort(dist_chunk, axis=1)[:, 1:nn+1] + start
        return indices_chunk

    n_samples = A.shape[1]
    nn_indices = np.empty((n_samples, nn), dtype=int)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        dist_chunk = pairwise_distances_chunked(
            A.T[start:end],
            A.T,
            metric='chebyshev',
            reduce_func=lambda D, start_chunk=start: process_chunk(D, start_chunk)
        )
        nn_indices[start:end] = next(dist_chunk)
    
    return nn_indices

def nn_pairs_heuristic(A: np.ndarray, nn: int, batch_size: int = 1000):
    A = A.T
    n_samples = A.shape[0]
    rangeLenA = np.arange(n_samples)
    np.random.shuffle(rangeLenA)
    APrime = A[rangeLenA]
    nns = np.empty((n_samples, nn), dtype=int)
    reverse_indices = np.zeros(n_samples, dtype=int)
    for i, idx in enumerate(rangeLenA):
        reverse_indices[idx] = i
    for idx_start in tqdm(range(0, n_samples, batch_size)):
        if (idx_start + batch_size * 2) > n_samples:
            nns[idx_start:] = rangeLenA[nn_pairs(APrime[idx_start:].T, nn) + idx_start]
            break
        idx_end = idx_start + batch_size + batch_size
        nns[idx_start:idx_end] = rangeLenA[nn_pairs(APrime[idx_start:idx_end].T, nn) + idx_start]
    nns = nns[reverse_indices]
    return nns

def build_nn_graph(nn_pairs):
    pairs = dict(zip(range(len(nn_pairs)), nn_pairs))
    G = nx.Graph(pairs)
    return G
