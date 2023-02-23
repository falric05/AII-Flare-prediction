import matplotlib.pyplot as plt
import numpy as np

figsize = (14, 4)

def plot_serie(X, labels=None, thr=None):
    """
    PLots one single time serie
    ## Params
    * `X` is a one dimensional serie
    * `thr` is a threshold with same dimension of X. By default is `None`
    """
    assert thr is None or len(X) == len(thr)
    t = np.arange(len(X))
    plt.figure(figsize=figsize)
    plt.plot(t, X)
    if not labels is None:
        plt.scatter(t[labels], X[labels], color='orange', alpha=0.8)
    plt.xlabel('t')
    plt.ylabel('X(t)')
    plt.legend()
    plt.show()


def plot_grid_datapoints(grid, params, run=0, sigma=0.5, theta=0.1, mu=1, delta=0.2, labels=None, variation=None, num_run=10):
    """
    Plots datapoints according to pre-defined parameters
    ## Params
    * `grid` grid variables
    * `params` dictionary of parameters
    * `run=0`
    * `sigma=0.5` 
    * `theta=0.1`
    * `mu=1`
    * `delta=0.2`
    * `variation` if specified plots light curves according to different values of this parameter
    """
    assert run in range(params['run'])
    assert variation in list(params.keys())+[None]
    assert labels is None or variation is None

    sigma = params['sigma'].index(sigma)
    theta = params['theta'].index(theta)
    mu = params['mu'].index(mu)
    delta = params['delta'].index(delta)
    plt.figure(figsize=figsize)
    if variation is None:
        Xs = grid[run, sigma, theta, mu, delta, :]
        t = np.arange(grid.shape[-1])
        plt.plot(t, Xs)
        if not labels is None:
            ys = labels[run, sigma, theta, mu, delta, :]
            plt.scatter(t[ys], Xs[ys], color='orange', alpha=0.8)
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