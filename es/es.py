import numpy as np
from agent import Agent
from typing import Callable
from pop import Population


def genetic_algorithm(f: Callable, n: int, children: int=1000, parents: int=100, x0=None):
    """Run genetic algorithm on n-d function f with optional initial guess x0.
    """
    if x0 is None:
        x0 = np.zeros(n)
    pop = Population([Agent(x=np.random.normal(x0, np.ones(n) * 2), f=f, id=i) for i in range(children)])
    for i in range(100):
        new_parents = pop.get_best(parents)
        if i % 5 == 0:
            new_parents = [p for p in new_parents if p.has_valid_hessian()]

        mean = np.mean([p.x for p in new_parents], axis=0)
        std = np.std([p.x for p in new_parents], axis=0)
        pop = Population(new_parents + [Agent(x=np.random.normal(mean, std), f=f, id=i) for i in range(children - parents)])
        print(f'Generation {i}')
        print(f'\tMean fitness: {pop.get_mean_fitness()}')
        print(f'\tStddev fitness: {pop.get_stddev_fitness()}')
        print()
    print('Final results')
    print(f'\tMean fitness: {pop.get_mean_fitness()}')
    print(f'\tStddev fitness: {pop.get_stddev_fitness()}')
    print(f'\tAverage x value: {np.mean([p.x for p in pop.get_best(parents)], axis=0)}')

def genetic_algorithm_covar(f: Callable, n: int, children: int=100, parents: int=3, x0=None):
    """Run genetic algorithm on n-d function f with optional initial guess x0. Uses covariance matrix optimization.
    """
    pass

if __name__ == "__main__":
    genetic_algorithm(lambda x: np.sum(x**2), 10, children=1000, parents=100, x0=np.ones(10))