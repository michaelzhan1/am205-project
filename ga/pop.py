import numpy as np

class Population:

    def __init__(self, agents):
        self.agents = agents
        self.eval_fitnesses()

    def eval_fitnesses(self):
        self.fits = np.asarray([a.calc_fitness() for a in self.agents], dtype=np.float32)

    def get_mean_fitness(self):
        return np.mean(self.fits)
    
    def get_stddev_fitness(self):
        np.std(self.fits)
