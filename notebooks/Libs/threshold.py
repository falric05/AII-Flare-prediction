import numpy as np
from itertools import product

def get_labels_physics(grid, params):
    thr = np.ones(grid.shape)
    for r in range(params['run']):
        for s, t, d, m in product(params['sigma'], params['theta'], 
                                    params['delta'], params['mu']):
            si = params['sigma'].index(s)
            mi = params['mu'].index(m)
    thr[r, si, :, mi, :, :] *= m * (1 + s)
    return grid > thr

def get_labels_KDE(grid, params):
    raise NotImplementedError()

def get_labels_quantile(grid, params):
    raise NotImplementedError()
