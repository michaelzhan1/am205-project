from newton.newton import newton
from es.es import evo_strat
from es_newton.hybrid import es_newton
from cmaes.cmaes import cma_evo_strat
from cmaes_newton.hybrid import cmaes_newton
import numpy as np
import scipy
from deap import benchmarks


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
    print('-------------------------')

    print(f"Newton's method results: {x_newton} after {count_newton} iterations")
    print(f"Evolutionary strategy results: {x_es.x} after {count_es} iterations")
    print(f"Newton-ES hybrid results: {x_es_newton} after {count_es_newton} iterations")
    print(f"CMA-ES results: {x_cmaes.x} after {count_cmaes} iterations")
    print(f"CMA-ES-Newton hybrid results: {x_cmaes_newton} after {count_cmaes_newton} iterations")
    print(f"Scipy results: {x_scipy.x} after {x_scipy.nit} iterations")
    

if __name__ == "__main__":
    main()
