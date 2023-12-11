import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from typing import Callable
from .agent import Agent
from .pop import Population

def plot_population(f: Callable, pop: Population, name: str = 'Untitled'):
    fig, ax = plt.subplots()
    #ax.view_init(elev=90, azim=-90, roll=0)

    x = np.arange(-5, 5, 0.25)
    x1 = np.arange(-5, 5, 0.25)
    x, x1 = np.meshgrid(x, x1)
    Z = 100 * (x * x - x1)**2 + (1. - x)**2
    #Z = x**2 + 2 * x1**2 - 0.3 * np.cos(3 * np.pi * x) - 0.4 * np.cos(4 * np.pi * x1) + 0.7
    ax.contourf(x, x1, Z)
    
    
    for agent in pop.agents:
        ax.scatter(agent.x[0], agent.x[1])
    #surf = ax.plot_surface(x, x1, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    

    plt.savefig(name)
    plt.close()
