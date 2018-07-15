import numpy as np
from skopt import gp_minimize

class SMBO:

    def __init__(self, train_data, validation_data):

        self.train_data = train_data
        self.validation_data = validation_data

        self.dimensions = 0

        pass

    def response(self, hyperparameters):

        pass

    def optimize(self, initial_hyperparams):

        #gp_minimize(func=self.response, [(-2.0, 2.0)], n_random_starts=3, n_calls=10)

        pass