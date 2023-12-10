import numpy as np
from agent import Agent
from typing import Callable


def genetic_algorithm(f: Callable, n: int, children: int=100, parents: int=3, x0=None):
    """Run genetic algorithm on n-d function f with optional initial guess x0.
    """
    if x0 is None:
        x0 = np.zeros(n)
    identity = np.eye(n)
    agents = [Agent(x=np.random.normal(0, 1, size=(1, 3)))]


def genetic_algorithm_covar(f: Callable, n: int, children: int=100, parents: int=3, x0=None):
    """Run genetic algorithm on n-d function f with optional initial guess x0. Uses covariance matrix optimization.
    """
    pass