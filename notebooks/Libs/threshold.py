import numpy as np
from itertools import product
from sklearn.neighbors import KernelDensity
import os
import pandas as pd

from Libs.config import data_folder

def get_labels_physic(grid, params, alpha=1, classification=False, override=True):
    """
    Compute the threshold by considering the parameters that generated the current light curve
    as `thr = μ + α*σ*μ`
    ## Params
    * `grid`: grid of light curves
    * `params`: params dictionary
    * `alpha`: weight coefficient of stimated stochastic component (α), by default is equal to 1
    """
    if classification:
        filename = 'classification_labels_physic'
    else:
        filename = 'labels_physic'
            
    if not os.path.exists(os.path.join(data_folder, filename+'.npy')) or override:
        thr = np.ones(grid.shape)
        for r in range(params['run']):
            for s, m in product(params['sigma'], params['mu']):
                si = params['sigma'].index(s)
                mi = params['mu'].index(m)
                thr[r, si, :, mi, :, :] *= m * (1 + alpha*s)
        pred = grid >= thr
        # store labels
        np.save(os.path.join(data_folder, filename), pred, allow_pickle=True)
    # load labels
    pred = np.load(os.path.join(data_folder, filename+'.npy'), allow_pickle=True)
    return pred

def get_labels_KDE(grid, params, quantile=0.99, classification=False, override=True):
    """
    Estimate the threshold following the KDE anomaly detection approach
    ## Params
    * `grid`: grid of light curves
    * `params`: params dictionary
    * `alpha`: weight coefficient of stimated stochastic component (α), by default is equal to 1
    """
    if classification:
        filename = 'classification_labels_anomaly_detection'
    else:
        filename = 'labels_anomaly_detection'
    
    if not os.path.exists(os.path.join(data_folder, filename+'.npy')) or override:
        flares = np.ones(grid.shape, dtype=bool)
        
        for s, t, d, m in product(params['sigma'], params['theta'], params['delta'], params['mu']):
            si = params['sigma'].index(s)
            ti = params['theta'].index(t)
            di = params['delta'].index(d)
            mi = params['mu'].index(m)

            l = []
            signals = []
            for r in range(params['run']):
                run = grid[r, si, ti, mi, di, :].copy()
                max_value = run.max()
                run = run / max_value
                q1 = np.quantile(run, 0.25) 
                q3 = np.quantile(run, 0.75) 
                sigma = run.std()
                len_run =  len(run)
                h = 0.9 * min(sigma, (q3-q1)/ 1.34) * len_run**(-0.2)
                kde = KernelDensity(kernel='gaussian', bandwidth=h)
                kde.fit(run.reshape(-1, 1))
                ldens = kde.score_samples(run.reshape(-1, 1)) # Obtain log probabilities
                index = np.arange(len(run))
                signal = -ldens
                # signal = pd.Series(index=index, data=-ldens)
                signals.append(signal)
                scores = kde.score_samples(run.reshape(-1, 1))
                l.append(np.quantile(scores, quantile))
        
            thr = np.mean(l)

            preds = []
            for r in range(params['run']):
                preds.append(signals[r] >= thr)
            flares[:, si, ti, mi, di, :] = np.array(preds)
            flares[:, si, ti, mi, di, :] = np.bitwise_and(flares[:, si, ti, mi, di, :], 
                                                          grid[:, si, ti, mi, di, :] > m)
        # store labels
        np.save(os.path.join(data_folder, filename), flares, allow_pickle=True)
    # load labels
    flares = np.load(os.path.join(data_folder, filename+'.npy'), allow_pickle=True)
    return flares


'''
def get_labels_KDE(grid, params, quantile=0.99, classification=False, override=True):
    """
    Estimate the threshold following the KDE anomaly detection approach
    ## Params
    * `grid`: grid of light curves
    * `params`: params dictionary
    * `alpha`: weight coefficient of stimated stochastic component (α), by default is equal to 1
    """
    if classification:
        filename = 'classification_labels_anomaly_detection'
    else:
        filename = 'labels_anomaly_detection'
    
    if not os.path.exists(os.path.join(data_folder, filename+'.npy')) or override:
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
        np.save(os.path.join(data_folder, filename), pred, allow_pickle=True)
    # load labels
    pred = np.load(os.path.join(data_folder, filename+'.npy'), allow_pickle=True)
    return pred
'''

def get_labels_quantile(grid, params, percentile=0.8, classification=False, override=True):
    if classification:
        filename = 'classification_labels_quantile'
    else:
        filename = 'labels_quantile'
    
    if not os.path.exists(os.path.join(data_folder, filename+'.npy')) or override:
        labels = np.copy(grid)
        for s, t, d, m in product(params['sigma'], params['theta'],params['delta'], params['mu']):
            si = params['sigma'].index(s)
            ti = params['theta'].index(t)
            di = params['delta'].index(d)
            mi = params['mu'].index(m)
            # stima del percentile
            values = np.sort(grid[:,si,ti,mi,di,:].flatten())
            # normalized_values = values/values[-1]
            treshold = values[int(percentile*params['run']*params['N'])] # 0.8 * 30 * 1000
            view = labels[:,si,ti,mi,di,:]
            view[view>treshold] = 1
            view[view<=treshold] = 0
        result = labels.astype('bool')
        np.save(os.path.join(data_folder, filename), result, allow_pickle=True)
    result = np.load(os.path.join(data_folder, filename+'.npy'), allow_pickle=True)
    
    return result

def get_labels_quantile_on_run(grid, params, percentile=0.8, classification=False, override=True):
    if classification:
        filename = 'classification_labels_quantile_on_run'
    else:
        filename = 'labels_quantile_on_run'
    
    if not os.path.exists(os.path.join(data_folder, filename+'.npy')) or override:
        labels = np.copy(grid)
        for ri in range(params['run']):
            for s, t, d, m in product(params['sigma'], params['theta'],params['delta'], params['mu']):
                si = params['sigma'].index(s)
                ti = params['theta'].index(t)
                di = params['delta'].index(d)
                mi = params['mu'].index(m)
                # stima del percentile
                values = np.sort(grid[ri,si,ti,mi,di,:])
                normalized_values = values/values[-1]
                treshold = normalized_values[int(percentile*params['N'])]
                view = labels[ri,si,ti,mi,di,:]
                view[view>treshold] = 1
                view[view<=treshold] = 0
        result = labels.astype('bool')
    result = np.load(os.path.join(data_folder, filename+'.npy'), allow_pickle=True)
    return result

def get_labels_mean_std(grid, params, classification=False):
    # l = 2*theta/((sigma)**2)
    # mean = l * mu / (l-1)
    # std = np.sqrt( l*(mu**2) / (l-2) )
    raise NotImplementedError()