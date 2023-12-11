import numpy as np
from typing import Callable
from .agent import Agent
from .pop import Population


def evo_strat(f: Callable, n: int, children: int=1000, parents: int=100, x0=None, tol=1e-10, display=True, max_iter:int = 100):
    """Run genetic algorithm on n-d function f with optional initial guess x0.
    """
    if x0 is None:
        x0 = np.zeros(n)
    pop = Population([Agent(x=np.random.normal(x0, np.ones(n) * 2), f=f, id=i) for i in range(children)])
    prev_mean = np.zeros(n)
    for i in range(max_iter):
        new_parents = pop.get_best(parents)
        if i % 5 == 0:
            new_parents = [p for p in new_parents if p.has_valid_hessian()]

        mean = np.mean([p.x for p in new_parents], axis=0)
        std = np.std([p.x for p in new_parents], axis=0)
        pop = Population(new_parents + [Agent(x=np.random.normal(mean, std), f=f, id=i) for i in range(children - parents)])
        if display:
            print(f'Generation {i}')
            print(f'\tMean fitness: {pop.get_mean_fitness()}')
            print(f'\tStddev fitness: {pop.get_stddev_fitness()}')
            print()
        if np.linalg.norm(mean - prev_mean) < tol:
            break
        prev_mean = mean
    else:
        print('FAILED TO HIT THRESHOLD!')
            
    if display:
        print('Final results')
        print(f'\tMean fitness: {pop.get_mean_fitness()}')
        print(f'\tStddev fitness: {pop.get_stddev_fitness()}')
        print(f'\tAverage x value: {np.mean([p.x for p in pop.get_best(parents)], axis=0)}')
    
    return pop.get_best(1)[0]

if __name__ == "__main__":
    evo_strat(lambda x: np.sum(x**2), 10, children=1000, parents=100, x0=np.ones(10))
