import numpy as np
from itertools import product

def get_labels_physics(grid, params):
    thr = np.ones(grid.shape)
    for s, m in product(params['sigma'], params['mu']):
        thr[:, params['sigma'].index(s), :, params['mu'].index(m), :, :] *= m * (1 + s)
    return grid > thr

def get_labels_KDE(grid, params):
    raise NotImplementedError()

def get_labels_quantile(grid, params):
    raise NotImplementedError()
