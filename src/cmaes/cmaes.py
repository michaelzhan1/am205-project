import numpy as np
from typing import List, Callable
import scipy
from es.agent import Agent
from es.pop import Population
from es.plot_pop import plot_population

def cma_evo_strat(f: Callable, n: int, children: int=1000, parents: int=100, x0=None, tol=1e-10, display=True, max_iter=100, name='undefined', plot=False):
    count = 0
    if x0 is None:
        x0 = np.zeros(n)
    alpha_sigma = 0.5
    alpha_cp = 1
    alpha_c1 = alpha_clambda = 0.3
    alpha_mu = 1
    covar = np.eye(n)
    p_sigma = np.zeros(n)
    sigma = 1
    p_c = np.zeros(n)
    pop = Population([Agent(x=np.random.multivariate_normal(x0, sigma ** 2 * covar), f=f, id=i) for i in range(children)])

    prev_mean = np.zeros(n)
    for i in range(max_iter):
        if plot:
            plot_population(f, pop, f'{str(i).zfill(2)}_cmaes.png', name)

        count += 1
        new_parents = pop.get_best(parents)
        if i % 5 == 0:
            new_parents = [p for p in new_parents if p.has_valid_hessian()]
        lam = len(new_parents)
        
        mean = prev_mean + alpha_mu * np.mean([p.x - prev_mean for p in new_parents], axis=0)
        p_sigma = (1 - alpha_sigma) * p_sigma + np.sqrt(alpha_sigma * (2 - alpha_sigma) * lam) * scipy.linalg.fractional_matrix_power(covar, -0.5) @ (mean - prev_mean) / sigma
        p_c = (1 - alpha_cp) * p_c + np.sqrt(alpha_cp * (2 - alpha_cp) * lam) * (mean - prev_mean) / sigma
        sigma = sigma * np.exp(alpha_sigma / 1.1 * (np.linalg.norm(p_sigma) / (np.sqrt(n)*(1-1/(4*n)+1/(21*n**2))) - 1))
        covar = (1 - alpha_clambda - alpha_c1) * covar + alpha_c1*p_c.reshape(-1, 1) @ p_c.reshape(1, -1) + alpha_clambda * np.mean([(p.x - prev_mean).reshape(-1, 1) @ (p.x - prev_mean).reshape(1, -1) for p in new_parents], axis=0) / sigma**2
        pop = Population(new_parents + [Agent(x=np.random.multivariate_normal(mean, sigma ** 2 * covar), f=f, id=i) for i in range(children - parents)])
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
    return pop.get_best(1)[0], count

