from newton.newton import newton
from es.es import evo_strat
from es_newton.hybrid import es_newton
from cmaes.cmaes import cma_evo_strat
from cmaes_newton.hybrid import cmaes_newton
import numpy as np
import scipy
from deap import benchmarks
from tabulate import tabulate


np.random.seed(1)


def main():
    f = lambda x: benchmarks.cigar(x)[0]
    x0 = np.array([1, 1])
    x_newton, count_newton = newton(f, x0)
    print('Newton done')
    x_es, count_es = evo_strat(f, 2, children=1000, parents=100, x0=x0, max_iter=200, display=False)
    print('ES done')
    x_es_newton, count_es_newton = es_newton(f, 2, children=1000, parents=100, x0=x0, display=False)
    print('ES-Newton done')
    x_cmaes, count_cmaes = cma_evo_strat(f, 2, children=1000, parents=100, x0=x0, max_iter=100, display=False)
    print('CMA-ES done')
    x_cmaes_newton, count_cmaes_newton = cmaes_newton(f, 2, children=1000, parents=100, x0=x0, display=False)
    print('CMA-ES-Newton done')
    x_scipy = scipy.optimize.minimize(f, x0, method='BFGS', tol=1e-12)
    print('Scipy done')
    print('-----------------------------------------------------------------------------')

    method_names = ['Newton', 'ES', 'ES-Newton', 'CMA-ES', 'CMA-ES-Newton', 'Scipy']
    x_values = list(map(list, [x_newton, x_es.x, x_es_newton, x_cmaes.x, x_cmaes_newton, x_scipy.x]))
    count_values = [count_newton, count_es, count_es_newton, count_cmaes, count_cmaes_newton, x_scipy.nit]
    headers = ['Method', 'x*', 'Iterations']
    print(tabulate(list(zip(method_names, x_values, count_values)), headers=headers, tablefmt='orgtbl'))
    

if __name__ == "__main__":
    main()
