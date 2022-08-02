from grpc import method_handlers_generic_handler
import numpy as np
from .expression import Expression
import time
from multiprocessing.pool import ThreadPool


class Dataset():

    # def normalize(x, lb=None, up=None):
    #     if lb is None:
    #         lb = x.min()
    #         up = x.max()
    #         return np.divide(np.subtract(x, lb), np.subtract(up, lb)), lb, up
    #     return np.divide(np.subtract(x, lb), np.subtract(up, lb))

    def __init__(self, n_functions, language, max_passes=0, model=None):
        self.dataset = []
        self.language = language
        self.max_passes = max_passes
        self.model = model

        import time
        start = time.time()
        count = 0

        while count != n_functions:
            row = {}  # To save the result
            row['n_obs'] = np.random.randint(10, 256)  # Number of observations
            row['Target Expression'] = Expression(language)  # Target Expression
            row['X_lower_bound'] = np.random.uniform(0.05, 6, size=language.max_variables)  # Lower bound of the variables
            row['X_upper_bound'] = [np.random.uniform(row['X_lower_bound'][i] + 1, 10) for i in range(language.max_variables)] # Upper bound of the variables
            row['X'] = np.concatenate([np.random.uniform(row['X_lower_bound'][i], row['X_upper_bound'][i], (row['n_obs'], 1)) for i in range(language.max_variables)], axis=1)
            # row['X_norm'] = Dataset.normalize(row['X'], lb=row['X_lower_bound'], up=row['X_upper_bound'])
            row['y'] = row['Target Expression'].evaluate(row['X'])
            miny = np.abs(row['y']).min()
            maxy = np.abs(row['y']).max()
            if (np.isnan(row['y']).any() or maxy > 1e5 or maxy < 1e-2 or maxy - miny == 0):
                continue
            # row['y_norm'], row['y_lower_bound'], row['y_upper_bound'] = Dataset.normalize(row['y'])
            self.dataset.append(row)
            count += 1

        print('Time intermediate: ', time.time() - start)
        pool = ThreadPool(processes=1)
        pool.map(self._add_prev_exprs, self.dataset)
        print('Time: ', time.time() - start)

    
    def _add_prev_exprs(self, row):
        row['Input Expressions'] = []
        row['y_preds'] = []
        if self.max_passes:
            iterations = np.random.randint(0, self.max_passes + 1)
            for _ in range(iterations):
                new_expression = Expression(self.language, model=self.model, prev_info=row)
                row['Input Expressions'].append(new_expression)
                y_pred = new_expression.evaluate(row['X'])
                if not (np.isnan(y_pred).any() or np.abs(y_pred).max() > 1e5 or np.abs(y_pred).min() < 1e-2):
                    row['y_preds'].append(new_expression.evaluate(row['X']))
                else:
                    break


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]