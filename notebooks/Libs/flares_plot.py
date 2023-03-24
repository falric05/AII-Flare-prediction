import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

figsize = (14, 4)

def plot_serie(X, labels=None, thr=None, title=''):
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
    plt.title(title)
    plt.show()


def plot_grid_datapoints(grid, params, run=0, sigma=0.5, theta=0.1, mu=1, delta=0.2, num_run=5, variation=None, labels=None, title='',
                        show=True):
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
    # assert labels is None or variation is None

    sigma = params['sigma'].index(sigma)
    theta = params['theta'].index(theta)
    mu = params['mu'].index(mu)
    delta = params['delta'].index(delta)
    plt.figure(figsize=figsize)
    plt.title(title)
    if variation is None:
        Xs = grid[run, sigma, theta, mu, delta, :]
        t = np.arange(grid.shape[-1])
        plt.plot(t, Xs)
        if not labels is None:
            ys = labels[run, sigma, theta, mu, delta, :]
            plt.scatter(t[ys], Xs[ys], color='orange', alpha=0.8)
    else:
        ys = np.zeros(grid.shape[-1], dtype='bool')
        if variation == 'theta':
            Xs = grid[run, sigma, :, mu, delta, :]
            if not labels is None:
                ys = labels[run, sigma, :, mu, delta, :]
        elif variation == 'sigma':
            Xs = grid[run, :, theta, mu, delta, :]
            if not labels is None:
                ys = labels[run, :, theta, mu, delta, :]
        elif variation == 'mu':
            Xs = grid[run, sigma, theta, :, delta, :]
            if not labels is None:
                ys = labels[run, sigma, theta, :, delta, :]
        elif variation == 'delta':
            Xs = grid[run, sigma, theta, mu, :, :]
            if not labels is None:
                ys = labels[run, sigma, theta, mu, :, :]
        elif variation == 'run':
            Xs = grid[:num_run, sigma, theta, mu, delta, :]
            if not labels is None:
                ys = labels[:num_run, sigma, theta, mu, delta, :]
        if variation != 'run':
            labels = [variation+' = '+str(p) for p in params[variation]]
        else:
            labels = [variation+' = '+str(p) for p in range(params[variation])]
        t = np.arange(grid.shape[-1])
        for X, y, l in zip(Xs, ys, labels):
            plt.plot(t, X, label = l)
            if not labels is None:
                plt.scatter(t[y], X[y], color='orange', alpha=0.8)
        plt.legend()
    plt.xlabel('t')
    plt.ylabel('X(t)')
    if show:
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
    
def plot_grid_PDF(grid, params, nbins= 10, run=0, sigma=0.5, theta=0.1, mu=1, delta=0.2, 
                  on_run=True, log_scale=True, dstep=1e-2, precision=1e-3, title=''):
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
    
    min_value, max_value = plot_histograms(grid, params, nbins=nbins, run=run, sigma=sigma, theta=theta, mu=mu, delta=delta,
                                           variation=None, on_run=on_run, normalize=True, log_scale=False, show=False)
    
    # theta, sigma, mu
    
    l = 2*theta/(sigma**2) # lambda
    values = np.arange(min_value, max_value, dstep)
    px = np.exp(-(l*mu/ values)) / (values**(l+2))
    
    k = 1/np.sum(px) # estimate of the normalization constant
    assert np.abs(1 - np.sum(k*px)) < precision # test of distribution
        
    plt.plot(values, k*px, label='Tavecchio PDF')
        
    if log_scale:
        plt.yscale('log')
        
    plt.xlabel('X')
    plt.ylabel('dP/dX')
        
    plt.legend()
    plt.title(title)
    plt.show()
    
    

def plot_histograms(grid, params, nbins=10, run=0, sigma=0.5, theta=0.1, mu=1, delta=0.2, variation=None, 
                    on_run=True, normalize=True, log_scale=True, title='', show=True):
    si = params['sigma'].index(sigma)
    ti = params['theta'].index(theta)
    mi = params['mu'].index(mu)
    di = params['delta'].index(delta)
    
    min_value = np.inf
    max_value = 0
    
    if variation is None:
        if on_run:
            datas = grid[run, si, ti, mi, di, :]
        else:
            datas = grid[:, si, ti, mi, di, :]
        hist, bin_edges = np.histogram(datas, bins=nbins)
        if bin_edges[0] < min_value:
            min_value = bin_edges[0]
        if bin_edges[-1] > max_value:
            max_value = bin_edges[-1]
        if normalize:
            hist = hist / np.sum(hist)
        x = [bin_edges[i//2+i%2] for i in range(2*nbins)]
        y = [hist[i//2] for i in range(2*nbins)]
        plt.plot(x,y, label='histogram')
    else:
        if variation=='sigma':
            for value in params[variation]:
                i = params[variation].index(value)
                if on_run:
                    datas = grid[run,i,ti,mi,di,:]
                else:
                    datas = grid[:,i,ti,mi,di,:]
                hist, bin_edges = np.histogram(datas, bins=nbins)
                if bin_edges[0] < min_value:
                    min_value = bin_edges[0]
                if bin_edges[-1] > max_value:
                    max_value = bin_edges[-1]
                if normalize:
                    hist = hist / np.sum(hist)
                x = [bin_edges[i//2+i%2] for i in range(2*nbins)]
                y = [hist[i//2] for i in range(2*nbins)]
                plt.plot(x,y, label='histogram '+variation+'= '+str(value))
        elif variation=='theta':
            for value in params[variation]:
                i = params[variation].index(value)
                if on_run:
                    datas = grid[run,si,i,mi,di,:]
                else:
                    datas = grid[:,si,i,mi,di,:]
                hist, bin_edges = np.histogram(datas, bins=nbins)
                if bin_edges[0] < min_value:
                    min_value = bin_edges[0]
                if bin_edges[-1] > max_value:
                    max_value = bin_edges[-1]
                if normalize:
                    hist = hist / np.sum(hist)
                x = [bin_edges[i//2+i%2] for i in range(2*nbins)]
                y = [hist[i//2] for i in range(2*nbins)]
                plt.plot(x,y, label='histogram '+variation+'= '+str(value))
        elif variation=='mu':
            for value in params[variation]:
                i = params[variation].index(value)
                if on_run:
                    datas = grid[run,si,ti,i,di,:]
                else:
                    datas = grid[:,si,ti,i,di,:]
                hist, bin_edges = np.histogram(datas, bins=nbins)
                if bin_edges[0] < min_value:
                    min_value = bin_edges[0]
                if bin_edges[-1] > max_value:
                    max_value = bin_edges[-1]
                if normalize:
                    hist = hist / np.sum(hist)
                x = [bin_edges[i//2+i%2] for i in range(2*nbins)]
                y = [hist[i//2] for i in range(2*nbins)]
                plt.plot(x,y, label='histogram '+variation+'= '+str(value))
        elif variation=='delta':
            for value in params[variation]:
                i = params[variation].index(value)
                if on_run:
                    datas = grid[run,si,ti,mi,i,:]
                else:
                    datas = grid[:,si,ti,mi,i,:]
                hist, bin_edges = np.histogram(datas, bins=nbins)
                if bin_edges[0] < min_value:
                    min_value = bin_edges[0]
                if bin_edges[-1] > max_value:
                    max_value = bin_edges[-1]
                if normalize:
                    hist = hist / np.sum(hist)
                x = [bin_edges[i//2+i%2] for i in range(2*nbins)]
                y = [hist[i//2] for i in range(2*nbins)]
                plt.plot(x,y, label='histogram '+variation+'= '+str(value))
        if show:
            plt.legend()
    if log_scale:
        plt.yscale('log')
    if show:
        plt.title(title)
        plt.show()
    return min_value, max_value


def histogram_next_given_actual(grid, params, nbins=10, run=0, sigma=0.5, theta=0.1, mu=1, delta=0.2, 
                                normalize=True, on_run=True, figsize=figsize, title=''):
    
    si = params['sigma'].index(sigma)
    ti = params['theta'].index(theta)
    mi = params['mu'].index(mu)
    di = params['delta'].index(delta)
    
    if on_run:
        A = grid[run, si, ti, mi, di, :]
        A = A[np.newaxis,:]
    else:
        A = grid[:, si, ti, mi, di, :]
    actual_df = pd.DataFrame(A[:,:params['N']-1].flatten(), columns=['Actual_value'])
    next_df = pd.DataFrame(A[:,1:].flatten(), columns=['Next_value'])
    actual_next_df = pd.concat([actual_df, next_df], axis=1)
    h, x_edges, y_edges = np.histogram2d(actual_next_df.Actual_value, actual_next_df.Next_value, bins=(nbins, nbins))
    
    if normalize:
        denominator = h.sum(axis=1)
        denominator[denominator==0] = 1
        h = h / denominator[:,np.newaxis]
    
    width = x_edges[1:]-x_edges[:-1]
    depth = y_edges[1:]-y_edges[:-1]

    x, y = np.meshgrid(x_edges[:-1], y_edges[:-1])
    x, y = x.ravel(), y.ravel()

    width, depth = np.meshgrid(width, depth)
    width, depth = width.ravel(), depth.ravel()

    top = h.ravel()
    bottom = np.zeros_like(top)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection = '3d')
    ax.bar3d(x, y, bottom, width, depth, top, shade=True)
    ax.set_title(title)
    ax.set_xlabel("X_Actual")
    ax.set_ylabel("X_Next")
    if normalize:
        ax.set_zlabel("P(X_Next|X_Actual)")
    else:
        ax.set_zlabel("Count(X_Next|X_Actual)")
    plt.show()

# def pdf(X, mu, theta, sigma):
#     lmbda = 2*theta/(sigma**2)
#     pX_notNorm = np.exp((-lmbda)*mu/X)/X**(lmbda+2)
#     px = pX_notNorm/np.linalg.norm(pX_notNorm)

#     return pX / np.max(pX)