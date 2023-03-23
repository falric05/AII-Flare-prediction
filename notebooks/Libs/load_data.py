import numpy as np
from itertools import product
import os
from tqdm import tqdm

from Libs.config import data_folder, default_data_folder
from Libs.threshold import get_labels_physic, get_labels_KDE, get_labels_quantile, get_labels_quantile_on_run

import pandas as pd
from sklearn.model_selection import train_test_split

def get_dataset_split(grid_X, grid_y, test_size = 0.2, val_size=0.2, window_size = 10, overlap_size = 0, label_treshold=1,
                      split_on_run=False, shuffle_run=False, shuffle_window=False, get_validation=False, random_state=42,
                      get_info=False, params=None):
    def build_df(X_configuration, y_configuration, window_size=window_size, overlap_size=overlap_size, label_treshold = label_treshold,
                 get_info=get_info, info_params=None):
        
        stride = window_size - overlap_size
        num_windows = (X_configuration.shape[-1]-window_size)//stride + 1

        windows = np.zeros((X_configuration.shape[0]*(num_windows-1),window_size))
        windows_label = np.zeros((y_configuration.shape[0]*(num_windows-1),window_size), dtype='bool')
        
        if get_info:
            info_params_copy = info_params.copy()
            info_params_copy['N'] = [info_params_copy['N']]
            info_labels = list(info_params_copy.keys())[:-1]
            info_configuration = [t for t in product(*info_params_copy.values())]
            infos = np.zeros((X_configuration.shape[0]*(num_windows-1),len(info_labels)))
            label_ranges = []
            window_ranges = []

        for i in range(X_configuration.shape[0]):
            if get_info:
                label_ranges = label_ranges + [(j, j+window_size-1) for j in range(stride, stride*num_windows, stride)]
                window_ranges = window_ranges + [(j, j+window_size-1) for j in range(0, stride*(num_windows-1), stride)]
                for j in range(len(info_labels)):
                    infos[i*(num_windows-1):(i+1)*(num_windows-1),j] = np.array([info_configuration[i][j]]*(num_windows-1))
                    
            tmp_windows = np.array([X_configuration[i,j:j+window_size] for j in range(0,stride*num_windows,stride)])
            tmp_windows_labels = np.array([y_configuration[i,j:j+window_size] for j in range(0,stride*num_windows,stride)])
            windows[i*(num_windows-1):(i+1)*(num_windows-1),:] = tmp_windows[:-1,:]
            windows_label[i*(num_windows-1):(i+1)*(num_windows-1),:] = tmp_windows_labels[1:,:]

        windows_label = np.sum(windows_label, axis=-1)
        windows_label[windows_label<label_treshold] = 0
        windows_label[windows_label>=label_treshold] = 1
        
        df = pd.DataFrame(windows, columns=[f't_{i}' for i in range(windows.shape[-1])])
        y_df = pd.DataFrame({'future_flare':windows_label})
        if get_info:
            info_df = pd.DataFrame(infos, columns = info_labels)
            label_range_df = pd.DataFrame({'label_range':label_ranges})
            window_range_df = pd.DataFrame({'window_range':window_ranges})
            df = pd.concat([info_df, df, window_range_df, label_range_df, y_df], axis=1)
            df = df.astype({'run':'int'})
        else:
            df = pd.concat([df, y_df], axis=1)

        return df
    
    if get_info:
        params_copy = params.copy()
        params_copy['run'] = list(range(params_copy['run']))
    else:
        params_copy = None
    
    if split_on_run:
        if get_info:
            params_train = params_copy.copy()
            params_test = params_copy.copy()
        else:
            params_train = None
            params_test = None
        if get_validation:
            if get_info:
                params_val = params_copy.copy()
            else:
                params_val = None
            train_size = 1-test_size-val_size
            run_val_index = int(train_size * grid_X.shape[0])
            run_test_index = int((train_size+val_size) * grid_X.shape[0])
        else:
            train_size = 1-test_size
            run_test_index = int(train_size * grid_X.shape[0])
        
        if shuffle_run:
            np.random.seed(random_state)
            run_perm = np.random.permutation(grid_X.shape[0])
        else:
            run_perm = np.arange(grid_X.shape[0])
            
        if get_validation:
            # Training set
            X_train_configuration = grid_X[run_perm[:run_val_index], :, :, :, :, :]
            y_train_configuration = grid_y[run_perm[:run_val_index], :, :, :, :, :]
            if get_info:
                params_train['run'] = run_perm[:run_val_index].tolist()
            # Validation set
            X_val_configuration = grid_X[run_perm[run_val_index:run_test_index], :, :, :, :, :]
            y_val_configuration = grid_y[run_perm[run_val_index:run_test_index], :, :, :, :, :]
            if get_info:
                params_val['run'] = run_perm[run_val_index:run_test_index].tolist()
        else:
            # Training
            X_train_configuration = grid_X[run_perm[:run_test_index], :, :, :, :, :]
            y_train_configuration = grid_y[run_perm[:run_test_index], :, :, :, :, :]
            if get_info:
                params_train['run'] = run_perm[:run_test_index].tolist()
        # Test set
        X_test_configuration = grid_X[run_perm[run_test_index:], :, :, :, :, :]
        y_test_configuration = grid_y[run_perm[run_test_index:], :, :, :, :, :]
        if get_info:
            params_test['run'] = run_perm[run_test_index:].tolist()
        
        
        # Training set
        X_train_configuration = X_train_configuration.reshape((np.prod(X_train_configuration.shape[:-1]),
                                                               X_train_configuration.shape[-1]))
        y_train_configuration = y_train_configuration.reshape((np.prod(y_train_configuration.shape[:-1]), 
                                                               y_train_configuration.shape[-1]))
        df_train = build_df(X_train_configuration, y_train_configuration, info_params=params_train)
        if shuffle_window:
            df_train = df_train.sample(frac=1, random_state=random_state)
        
        
        
        # Validation set
        if get_validation:
            X_val_configuration = X_val_configuration.reshape((np.prod(X_val_configuration.shape[:-1]), 
                                                               X_val_configuration.shape[-1]))
            y_val_configuration = y_val_configuration.reshape((np.prod(y_val_configuration.shape[:-1]),
                                                               y_val_configuration.shape[-1]))
            df_val = build_df(X_val_configuration, y_val_configuration, info_params=params_val)
            if shuffle_window:
                df_val = df_val.sample(frac=1, random_state=random_state)
            
        # Test set 
        X_test_configuration = X_test_configuration.reshape((np.prod(X_test_configuration.shape[:-1]), 
                                                             X_test_configuration.shape[-1]))
        y_test_configuration = y_test_configuration.reshape((np.prod(y_test_configuration.shape[:-1]), 
                                                             y_test_configuration.shape[-1]))
        df_test = build_df(X_test_configuration, y_test_configuration, info_params=params_test)
        if shuffle_window:
            df_test = df_test.sample(frac=1, random_state=random_state)
    else:
        X_configuration = grid_X.reshape((np.prod(grid_X.shape[:-1]), grid_X.shape[-1]))
        y_configuration = grid_y.reshape((np.prod(grid_y.shape[:-1]), grid_y.shape[-1]))
        df = build_df(X_configuration, y_configuration, info_params=params_copy)
        
        if shuffle_window:
            df = df.sample(frac=1, random_state=random_state)
        
        if get_validation:
            train_size = 1-test_size-val_size
            window_val_index = int(train_size * df.shape[0])
            window_test_index = int((train_size+val_size) * df.shape[0])
        else:
            train_size = 1-test_size
            window_test_index = int(train_size * df.shape[0])
        if get_validation:
            df_train = df.iloc[:window_val_index,:]
            df_val = df.iloc[window_val_index:window_test_index,:]
            df_test = df.iloc[window_test_index:,:]
        else:
            df_train, df_test = df.iloc[:window_test_index,:], df.iloc[window_test_index:,:]
    
    if get_validation:
        return df_train, df_val, df_test
    else:
        return df_train, df_test


class DataLoader():
    params = {}
    standard_index = None
    grid = None

    def __closed_form_method(self, theta, mu, sigma, delta_t, x0=1, N=10000):
        X = np.zeros(N)
        X[0] = x0
        W = np.random.normal(0, 1, size=N)
        W[0] = 0
        for i in range(N-1):
            X[i+1] = X[i] + theta*(mu - X[i])*delta_t + sigma*W[i]*X[i]*np.sqrt(delta_t) + 0.5*(sigma**2)*X[i]*delta_t*(W[i]**2 - 1)
        return X
    
    
    def __init__(self, run=1000, N=1000, s=0.5, t=0.1, d=0.2, m=1, 
                 labelling_method=get_labels_physic, override=True, folder=default_data_folder):
        """
        ## Params
        * `override`: if True it will generate new data and labels and save them in the folder specified, 
                      the name of the labels file depends on the labelling method used
                      if False it will try to load both the data and the labels
        * `folder`: specifies the folder where to save/load the data
        * `labelling_method`: specifies the method for labelling the data, if None only the data will be generated
        """
        self.params['run'] = run
        self.params['sigma'] = s if type(s)==list else [s]
        self.params['theta'] = t if type(t)==list else [t]
        self.params['mu']    = m if type(m)==list else [m]
        self.params['delta'] = d if type(d)==list else [d]
        self.params['N'] = N
        self.override = override
        self.folder = folder
        self.labelling_method = labelling_method
        grid_path = os.path.join(self.folder, 'grid')
        
        if (not os.path.exists(os.path.join(grid_path+'.npy'))) or self.override:
            self.grid = np.zeros((self.params['run'], 
                                  len(self.params['sigma']),
                                  len(self.params['theta']),
                                  len(self.params['mu']),
                                  len(self.params['delta']),
                                  self.params['N']))
            for r in tqdm(range(self.params['run'])):
                for s, t, m, d in product(self.params['sigma'], self.params['theta'], self.params['mu'], self.params['delta']):
                    si = self.params['sigma'].index(s)
                    ti = self.params['theta'].index(t)
                    mi = self.params['mu'].index(m)
                    di = self.params['delta'].index(d)
                    self.grid[r,si,ti,mi,di,:] = self.__closed_form_method(t, m, s, d, N=self.params['N'])
            # store grid data
            np.save(grid_path, self.grid, allow_pickle=True)
        self.grid = np.load(grid_path+'.npy', allow_pickle=True)
        if self.labelling_method is not None:
            self.labels = self.labelling_method(self.grid, self.params, override=self.override, folder=self.folder)

    def get_grid(self):
        """
        Return grid
        """
        if self.labelling_method is not None:
            return self.grid, self.labels
        else:
            return self.grid
    
    def get_params(self):
        """
        Return params dictionary
        """
        return self.params
    
    def get_indexes(self, s=0.5, t=0.1, d=0.2, m=1):
        """
        Return values indexes
        """
        idx = (self.params['sigma'].index(s), self.params['theta'].index(t),
               self.params['mu'].index(m), self.params['delta'].index(d))
        return idx
    
    def get_configuration(self, s=0.5, t=0.1, d=0.2, m=1):
        """
        Return values configuration
        """
        idx = self.get_indexes(s=s,t=t,d=d,m=m)
        X_configuration = self.grid[:,idx[0],idx[1],idx[2],idx[3],:].reshape((self.grid.shape[0],1,1,1,1,self.grid.shape[-1]))
        if self.labelling_method is not None:
            y_configuration = self.labels[:,idx[0],idx[1],idx[2],idx[3],:].reshape((self.labels.shape[0],1,1,1,1,self.labels.shape[-1]))
            return X_configuration, y_configuration
        else:
            return X_configuration