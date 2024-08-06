from dataclasses import dataclass
from typing import Callable, Iterable, Union, Literal, Type
from ilupp import _BaseWrapper

@dataclass(frozen=True)
class SolverConfig:
    mu: float = 1
    gamma: float = 0.75
    tol: float = 1e-6 
    criterions: Iterable[Callable] = None
    armijo_alpha: float = 1
    armijo_sigma: float = 0.25
    armijo_beta: float = 0.75
    armijo_iter: int = 10
    mu_update_tol: float = 2
    mu_update_tol_decay: float = 0.9
    max_iter: int = 1000
    mu_min: float = 1e-5
    mu_max: float = 1e5
    cgtol_tau: float = 0.618
    cgtol_default: float = 1e-5
    parallel: bool = False
    verbose: bool = True
    preconditioner: Union[Literal['auto', 'fixed'], Type[_BaseWrapper]] = 'auto'
