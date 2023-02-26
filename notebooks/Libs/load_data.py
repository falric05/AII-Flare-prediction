import numpy as np
from itertools import product
import os
from pathlib import Path
from tqdm import tqdm

from Libs.config import data_folder

class DataLoader():
    params = {
        'run': 30,
        'sigma': [0.3, 0.4, 0.5, 0.6],
        'theta': [0.01, 0.1, 0.5, 3],
        'mu': [0.8, 0.9, 1, 1.1],
        'delta': [0.01,0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7],
        'N': 1000
    }
    standard_index = None
    grid = None

    def __closed_form_method(self, theta, mu, sigma, delta_t, x0=1, N=1000):
        X = np.zeros(N)
        X[0] = x0
        W = np.random.normal(0, 1, size=N)
        W[0] = 0
        for i in range(N-1):
            X[i+1] = X[i] + theta*(mu - X[i])*delta_t + sigma*W[i]*X[i]*np.sqrt(delta_t) + 0.5*(sigma**2)*X[i]*delta_t*(W[i]**2 - 1)
        return X

    def __init__(self):
        grid_path = os.path.join(data_folder, 'grid')

        if not(Path(grid_path+'.npy').is_file()):
            self.grid = np.zeros((self.params['run'], 
                                  len(self.params['sigma']),
                                  len(self.params['theta']),
                                  len(self.params['mu']),
                                  len(self.params['delta']),
                                  self.params['N']))
            for r in tqdm(range(self.params['run'])):
                for s, t, d, m in product(self.params['sigma'], self.params['theta'], self.params['delta'], self.params['mu']):
                    si = self.params['sigma'].index(s)
                    ti = self.params['theta'].index(t)
                    di = self.params['delta'].index(d)
                    mi = self.params['mu'].index(m)
                    self.grid[r,si,ti,mi,di,:] = self.__closed_form_method(t, m, s, d)
            # store grid data
            np.save(grid_path, self.grid, allow_pickle=True)
        self.grid = np.load(grid_path+'.npy', allow_pickle=True)
        # takes indeces of preferred parameters
        self.standard_index = (self.params['sigma'].index(0.5), self.params['theta'].index(0.01),
                               self.params['delta'].index(0.2), self.params['mu'].index(1))
    
    def get_grid(self):
        """
        Return grid
        """
        return self.grid
    
    def get_params(self):
        """
        Return params dictionary
        """
        return self.params
    
    def get_standard_values(self, run=0):
        """
        Return standard values
        """
        return self.grid[run, self.standard_index[0], self.standard_index[1], self.standard_index[2], self.standard_index[3], :]
    
    def get_standard_indexes(self):
        """
        Return standard values indeces
        """
        return self.standard_index