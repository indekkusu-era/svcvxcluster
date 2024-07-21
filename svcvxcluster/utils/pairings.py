import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KDTree
import networkx as nx

def nn_pairs(A: np.ndarray, nn: int):
    inp = A.T
    tree = KDTree(inp, metric='chebyshev')
    ind = tree.query(inp, nn + 1, return_distance=False)
    return ind[:, 1:]

def build_nn_graph(nn_pairs):
    pairs = dict(zip(range(len(nn_pairs)), nn_pairs))
    G = nx.Graph(pairs)
    return G
