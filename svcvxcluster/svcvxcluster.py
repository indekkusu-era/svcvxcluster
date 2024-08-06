import networkx as nx
import numpy as np
from dataclasses import asdict
from typing import Union, Literal, Optional
from .solvers import SGS_ADMM, Thread_SSNAL
from .solvers.solver_config import SolverConfig
from .utils import nn_pairs, auto_select_params, build_nn_graph
from .postprocessing import clusters, build_postprocessing_graph

class SvCvxCluster():
    def __init__(self, nn: Union[int, Literal['precomputed']], 
                 eps: float = None, C: float = None,
                 alpha: float = None, alpha_prime: float = None,
                 solver_warm_start=SGS_ADMM, solver=Thread_SSNAL,
                 warm_start_solver_config: Optional[SolverConfig] = None,
                 solver_config: Optional[SolverConfig] = None):
        assert not (eps is None or C is None) or not (alpha is None or alpha_prime is None), "you must specify either eps, C or alpha, alpha_prime. If you specify both, eps and C will be selected"
        self._eps = eps
        self._C = C
        self._alpha = alpha
        self._alpha_prime = alpha_prime
        self._nn = nn
        self._warm_start_solver_config = SolverConfig(max_iter=50, gamma=1, tol=1e-4)\
            if warm_start_solver_config is None else warm_start_solver_config
        self._solver_config = SolverConfig() if solver_config is None else solver_config
        self._solver_warm_start = solver_warm_start
        self._solver = solver
        self.Xbar_ = None
        self.Z_ = None
    
    def calc_graph(self, A):
        return build_nn_graph(nn_pairs(A, self._nn))
    
    @property
    def incidence_matrix(self):
        return nx.incidence_matrix(self._graph, oriented=True)

    def fit(self, X: np.ndarray, y=None, graph: nx.Graph = None, warm_start=False, X0=None, Z0=None):
        assert not (self._nn == 'precomputed') or (graph is not None)
        self._X = X.T
        self._graph = self.calc_graph(self._X) if (graph is None) else graph
        if self._eps is None or self._C is None:
            self._eps, self._C = auto_select_params(self._X.T, self._nn, 
                                                    self.incidence_matrix, self._alpha, self._alpha_prime)
        dict_solver_cfg = asdict(self._solver_config)
        warm_start_cfg = asdict(self._warm_start_solver_config)
        if warm_start:
            xbar0, Z0 = self._solver_warm_start(self._X, self._eps, self._C, self._graph, 
                                            **warm_start_cfg)
        else:
            xbar0 = X0 if X0 is None else X0.copy()
            Z0 = Z0 if Z0 is None else Z0.copy()
        xbar, Z = self._solver(self._X, self._eps, self._C, self._graph, 
                               X0=xbar0, Z0=Z0, 
                               **dict_solver_cfg)
        self.Xbar_ = xbar.copy()
        self.Z_ = Z.copy()
        self.post_processing_graph_ = build_postprocessing_graph(self.Xbar_, self.incidence_matrix, max(self._eps, 1e-4))
        self.labels()
        return self
    
    def labels(self, thresh=0.05):
        npts = self.Xbar_.shape[1]
        self.labels_ = np.zeros(npts) - 1
        cluster_threshold = thresh * npts
        i = 0
        for cluster in clusters(self.post_processing_graph_):
            if len(cluster) < cluster_threshold: continue
            self.labels_[list(cluster)] = i
            i += 1
        return self.labels_

class Norm1CvxCluster(SvCvxCluster):
    def __init__(self, 
                 nn: Union[int, Literal['precomputed']], gamma: float = None,
                 solver_warm_start=SGS_ADMM, solver=Thread_SSNAL,
                 warm_start_solver_config: Optional[SolverConfig] = None,
                 solver_config: Optional[SolverConfig] = None):
        super().__init__(nn, 0, gamma, solver_warm_start=solver_warm_start, 
                         solver=solver, warm_start_solver_config=warm_start_solver_config, solver_config=solver_config)

