import pandas as pd
import os
import torch
import torch.nn as nn
import numpy as np

from Autoencoder import Autoencoder
from KalmanFilter import KalmanFilter
from sklearn.preprocessing import MinMaxScaler
from torch.utils import data

class TimeSeriesDataset(data.Dataset):
    def __init__(self, data, F=None, B=None, H=None, Q=None, R=None, P=None, x0=None):
        self.X = data
        self.kf = KalmanFilter(F, B, H, Q, R, P, x0)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        ############## TO DO ##############
        # implement KalmanFilter
        # self.X[index] is a rolling window of log return
        # please let them filtered by KalmanFilter
        ###################################
        
        return self.X[index]


class PretrainedAEModel:
    '''This class will pretrained the Autoencoder for the agent'''
    def __init__(self, data_path=None, rolling_window=60, F=None, B=None, H=None, Q=None, R=None, P=None, x0=None):
        self.rolling_window = rolling_window
        high = self._data_preprocessing(pd.read_csv(os.path.join(data_path, 'High.csv'), index_col=0, parse_dates=True))
        low = self._data_preprocessing(pd.read_csv(os.path.join(data_path, 'Low.csv'), index_col=0, parse_dates=True))
        close = self._data_preprocessing(pd.read_csv(os.path.join(data_path, 'Close.csv'), index_col=0, parse_dates=True))
        df = np.concatenate([high, low, close], axis=1)
        
        dataset = TimeSeriesDataset(df, F, B, H, Q, R, P, x0)
        self.dataloader = data.DataLoader(dataset, batch_size=64, num_workers=6) # TO DO: train data

    def _data_preprocessing(self, df):
        '''preprocess the price data into log return'''
        df = (np.log(df) - np.log(df.shift(1))).dropna()
        df = self._split_sequence(df)
        
        return np.expand_dims(np.transpose(df, (0, 2, 1)), axis=1)

    def _split_sequence(self, df):
        x = list()
        for i in range(len(df)):
            end_ix = i + self.rolling_window
            if end_ix > len(df)-1:
                break
            seq_x = df[i:end_ix]
            x.append(seq_x)
        return np.array(x)
    


class LFSS:
    '''LFSS module for the agent'''
    def __init__(self, data_path=None, rolling_window=60, F=None, B=None, H=None, Q=None, R=None, P=None, x0=None):
        self.model = PretrainedAEModel(data_path, rolling_window, F, B, H, Q, R, P, x0)
        pass