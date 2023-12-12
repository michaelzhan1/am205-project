from newton.newton import newton
from typing import Callable
from es.es import evo_strat
import numpy as np
from deap import benchmarks

def es_newton(f: Callable, n: int, children: int=1000, parents: int=100, x0=None, tol=1e-12, display=True, max_iter:int = 100, name='undefined'):
    x_es, es_count = evo_strat(f, n, children, parents, x0, tol, display, max_iter)
    init_newt = x_es.x
    x_newton, newton_count = newton(f, init_newt, 1e-15, max_iter)
    return x_newton, newton_count + es_count
