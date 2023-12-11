import numpy as np
from typing import List
from .agent import Agent

class Population:

    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.sort_fitness()
        
    def sort_fitness(self):
        self.agents.sort(key=lambda x: x.fitness)

    def get_best(self, n=100):
        return self.agents[:n]
    
    def get_mean_fitness(self):
        return np.mean([a.fitness for a in self.agents])
    
    def get_stddev_fitness(self):
        return np.std([a.fitness for a in self.agents])
