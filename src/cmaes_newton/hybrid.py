from newton.newton import newton
from typing import Callable
from cmaes.cmaes import cma_evo_strat
import numpy as np
from deap import benchmarks



def cmaes_newton(f: Callable, n: int, children: int=1000, parents: int=100, x0=None, tol=1e-12, display=True, max_iter:int = 100, name='undefined'):
    x_cmaes, cmaes_count = cma_evo_strat(f, n, children, parents, x0, tol, display, max_iter)
    init_newt = x_cmaes.x
    x_newton, newton_count = newton(f, init_newt, 1e-14, max_iter)
    return x_newton, newton_count + cmaes_count
