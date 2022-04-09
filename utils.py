import torch
import os
import pandas as pd
import numpy as np

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=20):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t/self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


# def three_dim_array(data_path):
#     '''this function will preprocess multiple pandas dataframe into 3D numpy array'''
    
#     data_list = os.listdir(data_path)
#     high_df = pd.DataFrame()
#     low_df = pd.DataFrame()
#     close_df = pd.DataFrame()
    
#     for i in data_list:
#         temp_df = pd.read_csv(f'data/{i}') # assume all the data start at the same date
#         high_df = pd.concat([high_df, temp_df['High']], axis=1, ignore_index=True)
#         low_df = pd.concat([low_df, temp_df['Low']], axis=1, ignore_index=True)
#         close_df = pd.concat([close_df, temp_df['Close']], axis=1, ignore_index=True)
    
#     close = close_df.to_numpy()
#     high = high_df.to_numpy()
#     low = low_df.to_numpy()
    
#     return np.stack((high, low, close))