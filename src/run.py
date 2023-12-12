from newton.newton import newton
from es.es import evo_strat
from es_newton.hybrid import es_newton
from cmaes.cmaes import cma_evo_strat
from cmaes_newton.hybrid import cmaes_newton
import numpy as np
import scipy
from deap import benchmarks
from tabulate import tabulate
import time
import pandas as pd


np.random.seed(1)


def main():
    f = lambda x: benchmarks.rosenbrock(x)[0]
    name = 'rosenbrock'
    x0 = np.array([1, 1])

    start_time = time.time()
    x_newton, count_newton = newton(f, x0, tol=1e-15)
    time_newton = time.time() - start_time
    print('Newton done')

    start_time = time.time()
    x_es, count_es = evo_strat(f, 2, children=1000, parents=100, x0=x0, max_iter=100, display=False, name=name)
    time_es = time.time() - start_time
    print('ES done')

    start_time = time.time()
    x_es_newton, count_es_newton = es_newton(f, 2, children=1000, parents=100, x0=x0, display=False, name=name)
    time_es_newton = time.time() - start_time
    print('ES-Newton done')

    start_time = time.time()
    x_cmaes, count_cmaes = cma_evo_strat(f, 2, children=1000, parents=100, x0=x0, max_iter=100, display=False, name=name)
    time_cmaes = time.time() - start_time
    print('CMA-ES done')

    start_time = time.time()
    x_cmaes_newton, count_cmaes_newton = cmaes_newton(f, 2, children=1000, parents=100, x0=x0, display=False, name=name)
    time_cmaes_newton = time.time() - start_time
    print('CMA-ES-Newton done')
    
    start_time = time.time()
    x_scipy = scipy.optimize.minimize(f, x0, method='BFGS', tol=1e-15)
    time_scipy = time.time() - start_time
    print('Scipy done')
    print('------------------------------------------------------------------------------------')

    method_names = ['Newton', 'ES', 'ES-Newton', 'CMA-ES', 'CMA-ES-Newton', 'Scipy']
    x_values = list(map(list, [x_newton, x_es.x, x_es_newton, x_cmaes.x, x_cmaes_newton, x_scipy.x]))
    count_values = [count_newton, count_es, count_es_newton, count_cmaes, count_cmaes_newton, x_scipy.nit]
    time_values = [time_newton, time_es, time_es_newton, time_cmaes, time_cmaes_newton, time_scipy]
    headers = ['Method', 'x*', 'Iterations', 'Time']
    print(tabulate(list(zip(method_names, x_values, count_values, time_values)), headers=headers, tablefmt='orgtbl'))
    df = pd.DataFrame({
        'Method': method_names,
        'x*': x_values,
        'Iterations': count_values,
        'Time': time_values
    })
    df.to_excel(f'{name}.xlsx')
    

if __name__ == "__main__":
    main()
