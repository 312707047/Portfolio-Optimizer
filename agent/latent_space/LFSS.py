import pandas as pd
import os
import torch
import torch.nn as nn
import numpy as np
import itertools

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
    def __init__(self,
                 data_path=None,
                 rolling_window=60,
                 F=None, B=None, H=None, Q=None, R=None, P=None, x0=None,
                 device=None):
        
        self.rolling_window = rolling_window
        self.device = device
        high = self._data_preprocessing(pd.read_csv(os.path.join(data_path, 'High.csv'), index_col=0, parse_dates=True))
        low = self._data_preprocessing(pd.read_csv(os.path.join(data_path, 'Low.csv'), index_col=0, parse_dates=True))
        close = self._data_preprocessing(pd.read_csv(os.path.join(data_path, 'Close.csv'), index_col=0, parse_dates=True))
        df = np.concatenate([high, low, close], axis=1)
        
        dataset = TimeSeriesDataset(df, F, B, H, Q, R, P, x0)
        self.dataloader = data.DataLoader(dataset, batch_size=64, num_workers=6) # TO DO: train data

        # model for training Autoencoder
        self.model = Autoencoder()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
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
    
    def train(self, loss_threshold=0.1):
        total_loss = []
        print('-------------Initialize Autoencoder-------------')
        for epoch in itertools.count():
            
            self.model.train()
            train_loss = []
            
            for batch in self.dataloader:
                
                x, y = batch, batch
                
                logits = self.model(x.to(self.device))
                loss = self.criterion(logits, y.to(self.device))
                
                L2_regularization = sum(p.pow(2.0).sum() for p in self.model.parameters())
                param_num = sum(p.numel() for p in self.model.parameters())
                loss += (0.001 / param_num) * L2_regularization
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss.append(loss.item())
            
            train_total_loss = sum(train_loss) / len(train_loss)
            total_loss.append(train_total_loss)
            
            if max(total_loss[-5:]) < loss_threshold:
                print('-------------Finish pretraining the model!-------------')
                torch.save(self.model.state_dict(), 'agent/latent_space/Autoencoder.ckpt')
                break
    
    def evaluate(self, model_path='agent/latent_space/Autoencoder.ckpt'):
        pass
    


class LFSS:
    '''LFSS module for the agent'''
    def __init__(self, data_path=None, rolling_window=60, F=None, B=None, H=None, Q=None, R=None, P=None, x0=None):
        self.model = PretrainedAEModel(data_path, rolling_window, F, B, H, Q, R, P, x0)
        pass