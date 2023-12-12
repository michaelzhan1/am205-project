import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from deap import benchmarks
from typing import Callable
from .agent import Agent
from .pop import Population


def griewank_arg0(sol):
    return benchmarks.griewank(sol)[0]

def ackley_arg0(sol):
    return benchmarks.ackley(sol)[0]

def bohachevsky_arg0(sol):
    return benchmarks.bohachevsky(sol)[0]

def rosenbrock_arg0(sol):
    return benchmarks.rosenbrock(sol)[0]

def plot_population(f: Callable, pop: Population, fig_title: str = 'Untitled', name='undefined'):
    fig, ax = plt.subplots()
    
    if name == 'griewank':
        X = np.arange(-50, 50, 0.5)
        Y = np.arange(-50, 50, 0.5)
        X, Y = np.meshgrid(X, Y)
        Z = np.fromiter(map(f, zip(X.flat,Y.flat)),  dtype=float, count=X.shape[0]*X.shape[1]).reshape(X.shape)
    elif name == 'ackley':
        X = np.arange(-30, 30, 0.5)
        Y = np.arange(-30, 30, 0.5)
        X, Y = np.meshgrid(X, Y)
        Z = np.fromiter(map(ackley_arg0, zip(X.flat,Y.flat)), dtype=float, count=X.shape[0]*X.shape[1]).reshape(X.shape)
    elif name == 'bohachevsky':
        X = np.arange(-15, 15, 0.5)
        Y = np.arange(-15, 15, 0.5)
        X, Y = np.meshgrid(X, Y)
        Z = np.fromiter(map(bohachevsky_arg0, zip(X.flat,Y.flat)), dtype=float, count=X.shape[0]*X.shape[1]).reshape(X.shape)
    elif name == 'rosenbrock':
        X = np.arange(-5, 5, 0.1)
        Y = np.arange(-5, 5, 0.1)
        X, Y = np.meshgrid(X, Y)
        Z = np.fromiter(map(rosenbrock_arg0, zip(X.flat,Y.flat)), dtype=float, count=X.shape[0]*X.shape[1]).reshape(X.shape)
    elif name == 'cigar':
        X = np.arange(-20, 20, 0.1)
        Y = np.arange(-20, 20, 0.1)
        X, Y = np.meshgrid(X,Y)
        Z = X**2 + 1e6 * Y**2

    ax.contourf(X, Y, Z)

    for agent in pop.agents:
        ax.scatter(agent.x[0], agent.x[1])
    
        
    plt.savefig(fig_title)
    plt.close()
