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
        self.scaler = MinMaxScaler()
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        ############## TO DO ##############
        # implement KalmanFilter
        # self.X[index] is a rolling window of log return
        # please filtered them with KalmanFilter
        ###################################
        
        class KalmanFilter(object):
            def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):
        
                if(F is None or H is None):
                    raise ValueError("Set proper system dynamics.")

                self.n = F.shape[1]
                self.m = H.shape[1]

                self.F = F
                self.H = H
                self.B = 0 if B is None else B
                self.Q = np.eye(self.n) if Q is None else Q
                self.R = np.eye(self.n) if R is None else R
                self.P = np.eye(self.n) if P is None else P
                self.x = np.zeros((self.n, 1)) if x0 is None else x0

            def predict(self, u = 0):
                self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
                self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
                return self.x

            def update(self, z):
                y = z - np.dot(self.H, self.x)
                S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
                K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
                self.x = self.x + np.dot(K, y)
                I = np.eye(self.n)
                self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
                    (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
    
        def filterresult(measurements):
        dt = 1.0/60
        F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
        H = np.array([1, 0, 0]).reshape(1, 3)
        Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
        R = np.array([0.5]).reshape(1, 1)

        kf = KalmanFilter(F = F, H = H, Q = Q, R = R)
        predictions = []

        for z in measurements:
            predictions.append(np.dot(H,  kf.predict())[0])
            kf.update(z)
    
        return predictions
        
        for index in range(len(self.X)):
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
        self.dataloader = data.DataLoader(dataset, batch_size=64, num_workers=6)

        # model for training Autoencoder
        self.model = Autoencoder()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        self.model_path = 'agent/latent_space/Autoencoder.ckpt'
        
    def _data_preprocessing(self, df):
        '''preprocess the price data into log return'''
        df = (np.log(df) - np.log(df.shift(1))).dropna()
        df = self._scaler(df)
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
    
    def _scaler(self, array):
        '''scale 3D array with MinMaxScaler'''
        for i in range(array.shape[0]):
            scaler = MinMaxScaler()
            array[i, :, :] = scaler.fit_transform(array[i, :, :]) 
    
    def train(self, loss_threshold=0.5):
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
                print(f'Total Training Epochs: {epoch}')
                torch.save(self.model.state_dict(), self.model_path)
                break
    
    def predict(self, observation):
        '''
        this will denoise the observation
        Arg:
            observation: state space from trading environment, shape(3*8*60)
        '''
        self.model.load_state_dict(torch.load(self.model_path))
        observation = self._scaler(np.transpose(observation))
        observation = np.expand_dims(observation, axis=0)
        observation = torch.tensor(observation, dtype=torch.float, device=self.device)
        
        ############## To Do ##############
        # apply KalmanFilter on observation
        ###################################
        
        self.model.eval()
        with torch.no_grad():
            denoised_obs = self.model(observation)
        
        denoised_obs = torch.squeeze(denoised_obs, dim=0) # shape(3*8*30)
        
        return denoised_obs


class LFSS:
    '''LFSS module for the agent'''
    def __init__(self, data_path=None, rolling_window=60, F=None, B=None, H=None, Q=None, R=None, P=None, x0=None):
        self.model = PretrainedAEModel(data_path, rolling_window, F, B, H, Q, R, P, x0)
        pass