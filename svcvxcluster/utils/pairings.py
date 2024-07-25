import numpy as np
import networkx as nx
import threading
from tqdm import tqdm
from sklearn.neighbors import KDTree
from concurrent.futures import ThreadPoolExecutor

def nn_pairs(A: np.ndarray, nn: int, batch_size=None):
    inp = A.T
    if batch_size is None:
        batch_size = min(10000, inp.shape[0] // 100)
    tree = KDTree(inp, metric='chebyshev')
    threads = []
    nns = np.zeros((inp.shape[0], nn), dtype=int)
    with ThreadPoolExecutor(max_workers=None) as executor:
        for i in np.arange(0, inp.shape[0], batch_size):
            threads.append(
                executor.submit(lambda inp, tree: tree.query(inp, k=nn, return_distance=False), 
                                inp[i:min(i+batch_size, inp.shape[0])], 
                                tree))
        
        for i, task in enumerate(threads):
            start = i * batch_size
            end = min((i+1) * batch_size, inp.shape[0])
            nns[start:end] = task.result()
    return nns

def build_nn_graph(nn_pairs):
    pairs = dict(zip(range(len(nn_pairs)), nn_pairs))
    G = nx.Graph(pairs)
    return G
