import gym
import os

class TradingEnv(gym.Env):
    def __init__(self, data_path, rolling_window=60, commission=0.01, balance=10000):
        
        self.data = data_path
        self.rolling_window = rolling_window
        self.balance = balance
        self.commission = commission
    
    def _data_preprocessing(self, data_path):
        '''It take the path of a folder that contains data, and transform the data into 3 dimensional arrays'''
        data_list = os.listdir(data_path)
        
        for data in data_list:
            pass
    
    def step(self, action):
        pass
    
    def reset(self):
        pass