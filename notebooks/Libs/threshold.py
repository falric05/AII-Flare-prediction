import numpy as np
from itertools import product
from sklearn.neighbors import KernelDensity
import os

from Libs.config import data_folder

def get_labels_physics(grid, params, alpha=1):
    """
    Compute the threshold by considering the parameters that generated the current light curve
    as `thr = μ + α*σ*μ`
    ## Params
    * `grid`: grid of light curves
    * `params`: params dictionary
    * `alpha`: weight coefficient of stimated stochastic component (α), by default is equal to 1
    """
    thr = np.ones(grid.shape)
    for r in range(params['run']):
        for s, m in product(params['sigma'], params['mu']):
            si = params['sigma'].index(s)
            mi = params['mu'].index(m)
            thr[r, si, :, mi, :, :] *= m * (1 + alpha*s)
    return grid >= thr

def get_labels_KDE(grid, params, quantile=0.99):
    """
    Estimate the threshold following the KDE anomaly detection approach
    ## Params
    * `grid`: grid of light curves
    * `params`: params dictionary
    * `alpha`: weight coefficient of stimated stochastic component (α), by default is equal to 1
    """
    if not os.path.exists(os.path.join(data_folder, 'labels_anomaly_detection.npy')):
        pred = np.ones(grid.shape, dtype=bool)

        for s, t, d, m in product(params['sigma'], params['theta'], params['delta'], params['mu']):
            si = params['sigma'].index(s)
            ti = params['theta'].index(t)
            di = params['delta'].index(d)
            mi = params['mu'].index(m)

            l = []
            for r in range(params['run']):
                Xr = grid[r, si, ti, mi, di, :].copy()
                # normalize data
                max_value = Xr.max()
                Xr = Xr / max_value
                # compute rule of thumb for bandwidth of KDE
                q1 = np.quantile(Xr, 0.25) 
                q3 = np.quantile(Xr, 0.75) 
                sigma = Xr.std()
                m =  len(Xr)
                h = 0.9 * min(sigma, (q3-q1)/ 1.34) * m**(-0.2)
                # fit the KDE model
                kde = KernelDensity(kernel='gaussian', bandwidth=h)
                kde.fit(Xr.reshape(-1, 1))
                # extract the log probabilities
                scores = kde.score_samples(Xr.reshape(-1, 1))
                index = np.arange(len(Xr))
                # build series with neg. prob.
                signal = -scores
                l.append(np.quantile(scores, quantile))
            thr = np.mean(l)
            pred[:, si, ti, mi, di] = signal >= thr
        # store labels
        np.save(os.path.join(data_folder, 'labels_anomaly_detection'), pred, allow_pickle=True)
    # load labels
    pred = np.load(os.path.join(data_folder, 'labels_anomaly_detection'+'.npy'), allow_pickle=True)
    return pred

def get_labels_quantile(grid, params):
    raise NotImplementedError()

def get_labels_mean_std(grid, params):
    # l = 2*theta/((sigma)**2)
    # mean = l * mu / (l-1)
    # std = np.sqrt( l*(mu**2) / (l-2) )
    raise NotImplementedError()