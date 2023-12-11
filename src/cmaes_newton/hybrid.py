from newton.newton import newton
from typing import Callable
from cmaes.cmaes import cma_evo_strat
import numpy as np
from deap import benchmarks


def cmaes_newton(f: Callable, n: int, children: int=1000, parents: int=100, x0=None, tol=1e-8, display=True, max_iter:int = 100):
    x_es = cma_evo_strat(f, n, children, parents, x0, tol, display, max_iter)
    init_newt = x_es.x
    return newton(f, init_newt, 1e-12, max_iter)
