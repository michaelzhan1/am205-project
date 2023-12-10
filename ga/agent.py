import numpy as np

class Agent:
    
    def __init__(self, x:np.array, id:int = -1, fitness:int = -1):
        self.id = id
        self.fitness = fitness
        self.x = x
    
    def calcFitness(self):
        raise NotImplementedError

    def __str__(self):
        return f'ID: {self.id} \n\tx: {self.x} \n\tfitness: {self.fitness}'
