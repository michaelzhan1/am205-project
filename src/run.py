from newton.newton import newton
from es.es import evo_strat
from es_newton.hybrid import es_newton
from cmaes.cmaes import cma_evo_strat
from cmaes_newton.hybrid import cmaes_newton
import numpy as np
from deap import benchmarks


def main():
    f = lambda x: benchmarks.bohachevsky(x)[0]
    x0 = np.array([1, 1])
    x_newton = newton(f, x0)
    print('Newton done')
    x_es = evo_strat(f, 2, children=1000, parents=100, x0=x0, max_iter=200, display=False)
    print('ES done')
    x_es_newton = es_newton(f, 2, children=1000, parents=100, x0=x0, display=False)
    print('ES-Newton done')
    x_cmaes = cma_evo_strat(f, 2, children=1000, parents=100, x0=x0, max_iter=100, display=False)
    print('CMA-ES done')
    x_cmaes_newton = cmaes_newton(f, 2, children=1000, parents=100, x0=x0, display=False)
    print('CMA-ES-Newton done')
    print('-------------------------')

    print(f"Newton's method results: {x_newton}")
    print(f"Evolutionary strategy results: {x_es.x}")
    print(f"Newton-ES hybrid results: {x_es_newton}")
    print(f"CMA-ES results: {x_cmaes.x}")
    print(f"CMA-ES-Newton hybrid results: {x_cmaes_newton}")
    

if __name__ == "__main__":
    main()
