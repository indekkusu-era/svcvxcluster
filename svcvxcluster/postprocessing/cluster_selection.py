import networkx as nx
import numpy as np

def build_postprocessing_graph(xbar, incidence_matrix, eps):
    BX = xbar @ incidence_matrix
    inf_norm = np.linalg.norm(BX, ord=np.inf, axis=0)
    new_inc_mat = incidence_matrix[:, np.where(inf_norm < eps)[0]]
    new_inc_mat.data = np.abs(new_inc_mat.data)
    return nx.Graph((new_inc_mat @ new_inc_mat.T).astype(int))

def clusters(postprocessing_graph):
    yield from nx.connected_components(postprocessing_graph)

