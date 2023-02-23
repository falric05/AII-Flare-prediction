import numpy as np
from itertools import product

def get_labels_physics(grid, params, alpha=1):
    thr = np.ones(grid.shape)
    for r in range(params['run']):
        for s, m in product(params['sigma'], params['mu']):
            si = params['sigma'].index(s)
            mi = params['mu'].index(m)
            thr[r, si, :, mi, :, :] *= m * (1 + alpha*s)
    return grid > thr, thr

def get_labels_KDE(grid, params):
    raise NotImplementedError()

def get_labels_quantile(grid, params):
    raise NotImplementedError()
