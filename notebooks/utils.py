import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import os
from pathlib import Path
from tqdm import tqdm

base_folder = os.path.join(os.getcwd(), '..')
figsize = (14, 4)


params = {
        'run': 30,
        'sigma': [0.3, 0.4, 0.5, 0.6],
        'theta': [0.01, 0.1, 0.5, 3],
        'mu': [0.8, 0.9, 1, 1.1],
        'delta': [0.01,0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7],
        'N': 1000
    }

def plot_grid_datapoints(grid, params, run=0, sigma=0.5, theta=0.1, mu=1, delta=0.2, variation=None, num_run=10):
    assert run in range(params['run'])
    assert variation in list(params.keys())+[None]
    sigma = params['sigma'].index(sigma)
    theta = params['theta'].index(theta)
    mu = params['mu'].index(mu)
    delta = params['delta'].index(delta)
    plt.figure(figsize=figsize)
    if variation is None:
        Xs = grid[run, sigma, theta, delta, :]
        plt.plot(Xs)
    else:
        if variation == 'theta':
            Xs = grid[run, sigma, :, mu, delta, :]
        elif variation == 'sigma':
            Xs = grid[run, :, theta, mu, delta, :]
        elif variation == 'mu':
            Xs = grid[run, sigma, theta, :, delta, :]
        elif variation == 'delta':
            Xs = grid[run, sigma, theta, :, mu, :]
        elif variation == 'run':
            Xs = grid[:num_run, sigma, theta, mu, delta, :]
        labels = [variation+' = '+str(p) for p in params[variation]]
        for X, l in zip(Xs, labels):
            plt.plot(X, label = l)
    plt.xlabel('t')
    plt.ylabel('X(t)')
    plt.legend()
    plt.show()

def plot_bidimensional_datapoints(array, params, name):
    params = params[name]
    labels = [name+' = '+str(p) for p in params]
    plt.figure(figsize=figsize)
    for data,label in zip(array,labels):
        plt.plot(data,label=label)
    plt.xlabel('t')
    plt.ylabel('X(t)')
    plt.legend()
    plt.show()

def load_grid():
    grid_path = os.path.join(base_folder, 'Data', 'grid')


    if not(Path(grid_path+'.npy').is_file()):
        grid = np.zeros((params['run'], 
                        len(params['sigma']),
                        len(params['theta']),
                        len(params['mu']),
                        len(params['delta']),
                        params['N']))
        for r in tqdm(range(params['run'])):
            for s, t, d, m in product(params['sigma'], params['theta'],params['delta'], params['mu']):
                # TODO: just in case add also mu 
                si = params['sigma'].index(s)
                ti = params['theta'].index(t)
                di = params['delta'].index(d)
                mi = params['mu'].index(m)
                grid[r,si,ti,mi,di,:] = closed_form_method(t,m,s,d)
        # store grid data
        np.save(grid_path, grid, allow_pickle=True)
    grid = np.load(grid_path+'.npy', allow_pickle=True)
    return grid