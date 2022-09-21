import torch
import numpy as np
from .expression import Expression
from copy import deepcopy
from tqdm.contrib.concurrent import thread_map


class Dataset():

    def __init__(self, n_functions, language):
        self.dataset = []
        self.language = language

        count = 0
        
        while count != n_functions:
            row = {}  # To save the result
            row['n_obs'] = np.random.randint(10, 256)  # Number of observations
            row['Target Expression'] = Expression(language)  # Target Expression
            row['X_lower_bound'] = np.random.uniform(0.05, 6, size=language.max_variables)  # Lower bound of the variables
            row['X_upper_bound'] = [np.random.uniform(row['X_lower_bound'][i] + 1, 10) for i in range(language.max_variables)] # Upper bound of the variables
            row['X'] = np.concatenate([np.random.uniform(row['X_lower_bound'][i], row['X_upper_bound'][i], (row['n_obs'], 1)) for i in range(language.max_variables)], axis=1)
            row['y'] = row['Target Expression'].evaluate(row['X'])
            miny, maxy = np.abs(row['y']).min(), np.abs(row['y']).max()
            if (np.isnan(row['y']).any() or maxy > 1e5 or maxy - miny == 0):
                continue
            self.dataset.append(row)
            count += 1

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]