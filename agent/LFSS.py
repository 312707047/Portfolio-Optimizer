import pandas as pd
import os
import torch
import torch.nn as nn
import numpy as np
import itertools

from pykalman import KalmanFilter
from networks.Denoise import Autoencoder
from sklearn.preprocessing import MinMaxScaler
from torch.utils import data


class TimeSeriesDataset(data.Dataset):
    def __init__(self, data_path, rolling_window=60):
        self.data_path = data_path
        self.rolling_window = rolling_window
        
        self.kf = KalmanFilter(transition_matrices = [1],
                               observation_matrices = [1],
                               initial_state_mean = 0,
                               initial_state_covariance = 1.5,
                               observation_covariance = 1.5,
                               transition_covariance = 1/30)
        
        high = self._data_preprocessing(pd.read_csv(os.path.join(data_path, 'High.csv'), index_col=0, parse_dates=True))
        low = self._data_preprocessing(pd.read_csv(os.path.join(data_path, 'Low.csv'), index_col=0, parse_dates=True))
        close = self._data_preprocessing(pd.read_csv(os.path.join(data_path, 'Close.csv'), index_col=0, parse_dates=True))
        self.df = np.concatenate([high, low, close], axis=1) # shape()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        for i in range(len(self.df[index])):
            for j in range(len(self.df[index][i])):
                a, _ = self.kf.filter(self.df[index][i][j])
                self.df[index][i][j] = np.squeeze(a)
                
        X = torch.tensor(self.df[index], dtype=torch.float)
        return X
       
    def _data_preprocessing(self, df):
        '''preprocess the price data into log return'''
        df = (np.log(df) - np.log(df.shift(1))).dropna()
        df = df.values
        df = self._split_sequence(df)
        df = self._scaler(df)
        
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
            return array

class PretrainedAEModel:
    '''This class will pretrained the Autoencoder for the agent'''
    def __init__(self,
                 data_path='data',
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        
        dataset = TimeSeriesDataset(data_path)
        self.dataloader = data.DataLoader(dataset, batch_size=32, num_workers=4)
        self.device = device
        self.model = Autoencoder().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model_path = 'agent/latent_space/Autoencoder.ckpt'
        
    def train(self, loss_threshold=0.3):
        total_loss = []
        print('-------------Initialize Autoencoder-------------')
        for epoch in itertools.count():
            
            self.model.train()
            train_loss = []
            
            for batch in self.dataloader:
                
                x, y = batch, batch
                
                logits = self.model(x.to(self.device, dtype=torch.float32))
                loss = self.criterion(logits, y.to(self.device, dtype=torch.float32))
                
                L2_regularization = sum(p.pow(2.0).sum() for p in self.model.parameters())
                param_num = sum(p.numel() for p in self.model.parameters())
                loss += (0.001 / param_num) * L2_regularization
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss.append(loss.item())
            
            train_total_loss = sum(train_loss) / len(train_loss)
            total_loss.append(train_total_loss)
            
            print(f'Epoch: {epoch} | Loss: {train_total_loss}')
            
            if max(total_loss[-5:]) < loss_threshold:
                print('-------------Finish pretraining the model!-------------')
                print(f'Total Training Epochs: {epoch}')
                torch.save(self.model.state_dict(), self.model_path)
                break
            
            if epoch+1 >= 100:
                print('-------------Finish pretraining the model!-------------')
                print(f'Total Training Epochs: {epoch}')
                torch.save(self.model.state_dict(), self.model_path)
                break
    
    def predict(self, observation):
        '''
        this will denoise the observation
        Arg:
            observation: state space from trading environment, shape(3*8*60) (Before kalmanfilter)
        '''
        self.model.load_state_dict(torch.load(self.model_path))
        model = self.model
        model = model.encoder
        if len(observation.shape) < 4:
            observation = torch.unsqueeze(observation, dim=0)
        
        self.model.eval()
        with torch.no_grad():
            denoised_obs = model(observation)
        
        return denoised_obs