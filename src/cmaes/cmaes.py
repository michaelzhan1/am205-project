import numpy as np
from typing import List, Callable
import scipy
from es.agent import Agent
from es.pop import Population


def cma_evo_strat(f: Callable, n: int, children: int=1000, parents: int=100, x0=None, tol=1e-10, display=True, max_iter=100):
    c_sigma = 1 / np.sqrt(n)
    c_c = 1 / np.sqrt(n)
    c_1 = 2 / n**2
    c_mu = 1 / n

    mean = x0 if x0 is not None else np.zeros(n)
    covar = np.eye(n)
    p_sigma = np.zeros(n)
    p_c = np.zeros(n)
    sigma = 1
    prev_mean = np.zeros(n)
    pop = Population([Agent(x=np.random.multivariate_normal(mean, sigma ** 2 * covar), f=f, id=i) for i in range(children)])
    for i in range(max_iter):
        new_parents = pop.get_best(parents)
        if i % 5 == 0:
            new_parents = [p for p in new_parents if p.has_valid_hessian()]
        lam = len(new_parents)
        prev_mean = mean
        mean = np.mean([p.x for p in new_parents], axis=0)
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(1-(1-c_sigma)**2)*np.sqrt(n)*scipy.linalg.fractional_matrix_power(covar, -0.5) @ (mean - prev_mean) / sigma
        new_sigma = sigma * np.exp(c_sigma * (np.linalg.norm(p_sigma) / (np.sqrt(n)*(1-1/(4*n)+1/(21*n**2))) - 1))
        p_c = (1 - c_c) + (1 if np.linalg.norm(p_sigma) <= 1.5 * np.sqrt(n) else 0) * (np.sqrt(1-(1-c_c)**2)*np.sqrt(n)*(mean - prev_mean) / sigma)
        covar = (1 - c_1 - c_mu) * covar + c_1*p_c.reshape(-1, 1) @ p_c.reshape(1, -1) + c_mu * np.mean([(p.x - prev_mean).reshape(-1, 1) @ (p.x - prev_mean).reshape(1, -1) for p in new_parents], axis=0) / sigma**2

        pop = Population(new_parents + [Agent(x=np.random.multivariate_normal(mean, new_sigma ** 2 * covar), f=f, id=i) for i in range(children - parents)])
        if display:
            print(f'Generation {i}')
            print(f'\tMean fitness: {pop.get_mean_fitness()}')
            print(f'\tStddev fitness: {pop.get_stddev_fitness()}')
            print()
        if np.linalg.norm(mean - prev_mean) < tol:
            break
        sigma = new_sigma
    else:
        print('FAILED TO HIT THRESHOLD!')
    
    if display:
        print('Final results')
        print(f'\tMean fitness: {pop.get_mean_fitness()}')
        print(f'\tStddev fitness: {pop.get_stddev_fitness()}')
        print(f'\tAverage x value: {np.mean([p.x for p in pop.get_best(parents)], axis=0)}')
    return pop.get_best(1)[0]
