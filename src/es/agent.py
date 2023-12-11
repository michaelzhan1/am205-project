import numpy as np
import numdifftools as nd
from typing import Callable

class Agent:
    def __init__(self, x:np.array, f: Callable, id:int = -1, fitness:int = -1):
        self.id = id
        self.f = f
        self.x = x
        self.fitness = self.calc_fitness()
    
    def calc_fitness(self):
        return self.f(self.x)
    
    def has_valid_hessian(self):
        hessian = nd.Hessian(self.f)
        return np.linalg.norm(hessian(self.x)) > 1e-10

    def __str__(self):
        return f'ID: {self.id} \n\tx: {self.x} \n\tfitness: {self.fitness}'
