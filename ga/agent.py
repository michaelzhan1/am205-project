import numpy as np
import numdifftools as nd
from typing import Callable

class Agent:
    def __init__(self, x:np.array, f: Callable, id:int = -1, fitness:int = -1):
        self.id = id
        self.fitness = fitness
        self.f = f
        self.x = x
    
    def calc_fitness(self):
        hessian = nd.Hessian(self.f)
        if np.norm(hessian(self.x)) < 1e-10:
            return 1e16
        else:
            return self.f(self.x)

    def __str__(self):
        return f'ID: {self.id} \n\tx: {self.x} \n\tfitness: {self.fitness}'
