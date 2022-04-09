import gym
import os
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, data_path, rolling_window=60, commission=0.01, balance=10000):
        
        self.data = data_path
        self.rolling_window = rolling_window
        self.balance = balance
        self.commission = commission
        self.data_num = os.listdir(data_path)

        self.action_space = gym.spaces.Box(
            0, 1, shape=(len(self.data_num) + 1,), dtype=np.float32)  # include cash
        
        spaces = {
            'portfolio': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.data_num, rolling_window, 3), dtype=np.float32),
            'covariance': gym.spaces.Box(low=-1.0, high=1.0, shape=(7, 7, 1), dtype=np.float32)
        }
        
        self.observation_space = gym.spaces.Dict(spaces)
        
        self.infos = []
        
    def _data_preprocessing(self, data_path):
        '''It take the path of a folder that contains data, and transform the data into 3 dimensional arrays'''
        data_list = os.listdir(data_path)
        
        for data in data_list:
            pass
    
    def step(self, action):
        pass
    
    def reset(self):
        pass