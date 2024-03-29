import numpy as np
from itertools import product
from sklearn.neighbors import KernelDensity
import os
from tqdm import tqdm

from Libs.config import data_folder, default_data_folder

def get_labels_physic(grid, params, alpha=1, override=True, folder=default_data_folder):
    """
    Compute the threshold by considering the parameters that generated the current light curve
    as `thr = μ + α*σ*μ`
    ## Params
    * `grid`: grid of light curves
    * `params`: params dictionary
    * `alpha`: weight coefficient of stimated stochastic component (α), by default is equal to 1
    """
    filename = 'labels_physic'
            
    if not os.path.exists(os.path.join(folder, filename+'.npy')) or override:
        thr = np.ones(grid.shape)
        for r in range(params['run']):
            for s, m in product(params['sigma'], params['mu']):
                si = params['sigma'].index(s)
                mi = params['mu'].index(m)
                thr[r, si, :, mi, :, :] *= m * (1 + alpha*s)
        pred = grid >= thr
        # store labels
        np.save(os.path.join(folder, filename), pred, allow_pickle=True)
    # load labels
    pred = np.load(os.path.join(folder, filename+'.npy'), allow_pickle=True)
    return pred

def get_labels_KDE(grid, params, quantile=0.99, override=True, folder=default_data_folder):
    """
    Estimate the threshold following the KDE anomaly detection approach
    ## Params
    * `grid`: grid of light curves
    * `params`: params dictionary
    * `quantile`: quantile which estimate the threshold for each run and configuration, default is 0.99
    """
    
    filename = 'labels_anomaly_detection'
    
    if not os.path.exists(os.path.join(folder, filename+'.npy')) or override:
        flares = np.ones(grid.shape, dtype=bool)
        # iterate over all parameters configurations
        configurations = tqdm(product(params['sigma'], params['theta'], params['mu'], params['delta']), 
                              total=len(params['sigma'])*len(params['theta'])*len(params['mu'])*len(params['delta']))
        for s, t, m, d in configurations:
            # get the index of each configuration 
            si = params['sigma'].index(s)
            ti = params['theta'].index(t)
            mi = params['mu'].index(m)
            di = params['delta'].index(d)

            l = []
            signals = []
            # get all the 30 threshold for a given configuration and store it in a list
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
                signal = -ldens
                signals.append(signal)
                scores = kde.score_samples(run.reshape(-1, 1))
                l.append(np.quantile(scores, quantile))
            # then compute the threshold mean for these 30 runs
            thr = np.mean(l)
            # and get the predictions 
            preds = []
            for r in range(params['run']):
                preds.append(signals[r] >= thr)
            flares[:, si, ti, mi, di, :] = np.array(preds)
            # KDE in principle could detect as anomaly values which are lower than mu, but we think
            # flares are just values higher than mu + a noise component, so we keep all anomalies which
            # are higher than mu
            flares[:, si, ti, mi, di, :] = np.bitwise_and(flares[:, si, ti, mi, di, :], 
                                                          grid[:, si, ti, mi, di, :] > m)
        # store labels
        np.save(os.path.join(folder, filename), flares, allow_pickle=True)
    # load labels
    flares = np.load(os.path.join(folder, filename+'.npy'), allow_pickle=True)
    return flares

def get_labels_quantile(grid, params, percentile=0.8, override=True, folder=default_data_folder):
    
    filename = 'labels_quantile'
    
    if not os.path.exists(os.path.join(folder, filename+'.npy')) or override:
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
        np.save(os.path.join(folder, filename), result, allow_pickle=True)
    result = np.load(os.path.join(folder, filename+'.npy'), allow_pickle=True)
    
    return result

def get_labels_quantile_on_run(grid, params, percentile=0.8, override=True, folder=default_data_folder):
    
    filename = 'labels_quantile_on_run'
    
    if not os.path.exists(os.path.join(folder, filename+'.npy')) or override:
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
        np.save(os.path.join(folder, filename), result, allow_pickle=True)
    result = np.load(os.path.join(folder, filename+'.npy'), allow_pickle=True)
    return result

def get_labels_mean_std(grid, params, override=True, folder=default_data_folder):
    # l = 2*theta/((sigma)**2)
    # mean = l * mu / (l-1)
    # std = np.sqrt( l*(mu**2) / (l-2) )
    raise NotImplementedError()