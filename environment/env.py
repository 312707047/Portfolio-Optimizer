import gym
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pykalman import KalmanFilter
import datetime as dt
from datetime import datetime

EPS = 1e-8

class TradingEnv(gym.Env):
    def __init__(self, data_path,
                 rolling_window=60,
                 commission=0.01,
                 steps=200,
                 start_date_index=None,
                 observation_features='Close'):
        '''
        Args:
            data_path: folder containing history data
            rolling_window: observation length for agent
            commission: just commission
            steps: steps in an episode
            start_date_index: the date index in the price array
            observation_features: choose how many features you'd like to input
                Close: close data only
                Three: including high, low, and close data
                All: Three + covariance matrix of high, low, and close data
        '''
        self.rolling_window = rolling_window
        self.commission = commission
        self.observation_features = observation_features
        self.data_path = data_path
        self.info_list = []
        
        self.kf = KalmanFilter(transition_matrices = [1],
                               observation_matrices = [1],
                               initial_state_mean = 1,
                               initial_state_covariance = 1.5,
                               observation_covariance = 1.5,
                               transition_covariance = 1/30)
        
        # read data
        self.close_prices = self._data_preprocessing(pd.read_csv(os.path.join(data_path, 'Close.csv'), index_col=0, parse_dates=True))
        self.close_obs = np.expand_dims(self.close_prices, 0) #shape(1, 1042, 8)
        if observation_features != 'Close':
            self.high_prices = self._data_preprocessing(pd.read_csv(os.path.join(self.data_path, 'High.csv'), index_col=0, parse_dates=True))
            self.low_prices = self._data_preprocessing(pd.read_csv(os.path.join(self.data_path, 'Low.csv'), index_col=0, parse_dates=True))
            
            self.high_obs = np.expand_dims(self.high_prices, 0)
            self.low_obs = np.expand_dims(self.low_prices, 0)
            
        self.tickers = self.close_prices.columns.to_list()
        self.tickers_num = len(self.tickers)
        self.dates = self.close_prices.index.values[1:]
        self.dates_num = self.dates.shape[0]
        # self.gain = np.hstack((np.ones((self.close_prices.shape[0]-1, 1)), self.close_prices.values[1:] / self.close_prices.values[:-1]))
        self.gain = self.close_prices.values[1:] / self.close_prices.values[:-1]
        
        self.info = []
        self.step_number = 0
        
        # Observation space and action space
        self.action_space = gym.spaces.Box(
            0, 1, shape=(self.tickers_num,), dtype=np.float32)  # include cash
        
        if observation_features == 'Close':
            
            spaces = {
                'portfolio': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.tickers_num, rolling_window), dtype=np.float32),
                'action': gym.spaces.Box(0, 1, shape=(self.tickers_num,), dtype=np.float32)
            }
        
        elif observation_features == 'Three':
            
            spaces = {
                'portfolio': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3, self.tickers_num, rolling_window), dtype=np.float32),
                'action': gym.spaces.Box(0, 1, shape=(self.tickers_num,), dtype=np.float32)
            }

        elif observation_features == 'All':
            spaces = {
                'portfolio': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3, self.tickers_num, rolling_window), dtype=np.float32),
                'action': gym.spaces.Box(0, 1, shape=(self.tickers_num,), dtype=np.float32),
                'covariance': gym.spaces.Box(low=-1.0, high=1.0, shape=(3, self.tickers_num, self.tickers_num), dtype=np.float32)
            }
            
        self.observation_space = gym.spaces.Dict(spaces)
        self.start_date_index = start_date_index
        self.steps = steps
        self.reset()
    
    def _data_preprocessing(self, df):
        '''preprocess the price data into log return'''
        return (np.log(df) - np.log(df.shift(1))).dropna()
    
    def _scaler(self, array):
        '''scale 3D array, shape(n, 60, 8) with MinMaxScaler'''
        for i in range(array.shape[0]):
            scaler = MinMaxScaler()
            array[i, :, :] = scaler.fit_transform(array[i, :, :])
            
        return array
    
    def _kalman(self, array):
        '''filter the noise of (3, 8, 60) array'''
        for i in range(len(array)):
            for j in range(len(array[i])):
                a, _ = self.kf.filter(array[i][j])
                array[i][j] = np.squeeze(a)
        
        return array

    def step(self, action):
        
        self.step_number += 1
        
        w1 = np.clip(action, a_min=0, a_max=1)
        # w1 = np.insert(w1, 0, np.clip(1 - w1.sum(), a_min=0, a_max=1))
        w1 = w1 / w1.sum()
        
        # 1. Calculate agent reward
        t = self.start_date_index + self.step_number
        y1 = self.gain[t]
        w0 = self.weights
        p0 = self.portfolio_value
        dw1 = (y1 * w0) / (np.dot(y1, w0)+EPS)
        mu1 = self.commission * (np.abs(dw1 - w1)).sum()
        p1 = p0 * (1 - mu1) * np.dot(y1, w1)
        p1 = np.clip(p1, 0, np.inf)
        rho1 = p1 / p0 - 1
        agent_return = np.log((p1+EPS)/(p0+EPS))

        # 2. Calculate return of same-weighted portfolio
        s_w0 = np.array([0.1/3, 0.1/3, 0.1/3, 0.9/5, 0.9/5, 0.9/5, 0.9/5, 0.9/5])
        # s_returns = self._data_preprocessing(self.close_prices.loc[:t]).sum().values
        # same_weighted_return = np.dot(s_returns, s_w)
        s_p0 = self.same_weighted_portfolio_value
        s_dw1 = (y1 * s_w0) / (np.dot(y1, s_w0)+EPS)
        s_mu1 = self.commission * (np.abs(s_dw1 - s_w0)).sum()
        s_p1 = s_p0 * (1 - s_mu1) * np.dot(y1, s_w0)
        s_p1 = np.clip(s_p1, 0, np.inf)
        same_weighted_return = np.log((s_p1+EPS)/(s_p0+EPS))
        
        reward = (agent_return - same_weighted_return) - 0.05 * max(w1)
        
        # save weights and portfolio value for next iteration
        self.weights = w1
        self.portfolio_value = p1
        self.same_weighted_portfolio_value = same_weighted_return
        
        # 3. Calculate MDD
        self.portfolio_value_list.append(p1)
        DD = min(self.portfolio_value_list) / max(self.portfolio_value_list) - 1
        
        
        # observe the next state
        t0 = t - self.rolling_window + 1
        
        if self.observation_features == 'Close':
            portfolio = self.close_obs[:, t0:t+1, :]
            portfolio = self._kalman(np.transpose(portfolio, [0, 2, 1]))
            
            observation = {'portfolio': portfolio,
                           'action': self.weights}
        
        elif self.observation_features == 'Three':
            portfolio = np.concatenate([self.high_obs, self.low_obs, self.close_obs], axis=0) # shape(3, 1042, 8)
            portfolio = portfolio[:, t0:t+1, :] # (3, 60, 8)
            portfolio = self._kalman(np.transpose(portfolio, [0, 2, 1]))
            
            observation = {'portfolio': portfolio,
                           'action': self.weights}
        
        elif self.observation_features == 'All':
            high_cov = np.expand_dims(np.cov(self.high_obs[:, t0:t+1, :][0].T), axis=0)
            low_cov = np.expand_dims(np.cov(self.low_obs[:, t0:t+1, :][0].T), axis=0)
            close_cov = np.expand_dims(np.cov(self.close_obs[:, t0:t+1, :][0].T), axis=0)
            covariance = np.concatenate([high_cov, low_cov, close_cov], axis=0) # shape(3, 8, 8)
            
            portfolio = np.concatenate([self.high_obs, self.low_obs, self.close_obs], axis=0) # shape(3, 1042, 8)
            portfolio = portfolio[:, t0:t+1, :] # (3, 60, 8)
            portfolio = self._kalman(np.transpose(portfolio, [0, 2, 1]))
            
            observation = {'portfolio':portfolio,
                           'action': self.weights,
                           'covariance':covariance}
        
        # info
        r = y1.mean()
        if self.step_number == 1:
            market_value = r
        else:
            market_value = self.info_list[-1]["market_value"] * r 
        info = {"reward": reward, "log_return": reward, "portfolio_value": p1, "return": r, "rate_of_return": rho1,
                "weights_mean": w1.mean(), "weights_std": w1.std(), "cost": mu1, 'date': self.dates[t],
                'steps': self.step_number, "market_value": market_value}
        self.info_list.append(info)
        
        # 4. Check limitation and done
        done = False
        if (self.step_number >= self.steps) or (p1 <= 0):
            done = True
        
        # Limitation 1: None of the asset should have higher ration than 65%
        for i in w1:
            if i > 0.65:
                done = True
                reward = -10
        
        # Limitation 2: Total ratio of cryptocurrency should not above 10%
        if sum(w1[:3]) > 0.1:
            done = True
            reward = -10
        
        # Reward shaping: MDD
        try:
            if min(self.DD) > DD:
                reward += -1
        except ValueError:
            pass
        
        if DD < 0:
            self.DD.append(DD)

        return observation, reward, done, info
    
    def reset(self):
        
        self.info = []
        self.weights = np.array([0.1/3, 0.1/3, 0.1/3, 0.9/5, 0.9/5, 0.9/5, 0.9/5, 0.9/5])
        # self.weights = np.insert(np.zeros(self.tickers_num), 0, 1.0)
        self.portfolio_value = 1.0
        self.same_weighted_portfolio_value = 1.0
        self.step_number = 0
        
        # Calculate MDD
        self.portfolio_value_list = [self.portfolio_value]
        self.DD = []
        
        self.steps = min(self.steps, self.dates_num - self.rolling_window - 1)
        
        if self.start_date_index is None:
            self.start_date_index = np.random.random_integers(self.rolling_window-1,
                                                              self.dates_num-self.steps-1)
        
        else:
            self.start_date_index = np.clip(self.start_date_index, a_min=self.rolling_window-1, a_max=self.dates_num-self.steps-1)
        
        t = self.start_date_index + self.step_number
        t0 = t - self.rolling_window + 1
        
        # Observation in different situations
        if self.observation_features == 'Close':
            portfolio = self.close_obs[:, t0:t+1, :]
            portfolio = self._kalman(np.transpose(portfolio, [0, 2, 1]))
            
            observation = {'portfolio': portfolio,
                           'action': self.weights}
        
        elif self.observation_features == 'Three':
            portfolio = np.concatenate([self.high_obs, self.low_obs, self.close_obs], axis=0) # shape(3, 1042, 8)
            portfolio = portfolio[:, t0:t+1, :] # (3, 60, 8)
            portfolio = self._kalman(np.transpose(portfolio, [0, 2, 1]))
            
            observation = {'portfolio': portfolio,
                           'action': self.weights}
        
        elif self.observation_features == 'All':
            high_cov = np.expand_dims(np.cov(self.high_obs[:, t0:t+1, :][0].T), axis=0)
            low_cov = np.expand_dims(np.cov(self.low_obs[:, t0:t+1, :][0].T), axis=0)
            close_cov = np.expand_dims(np.cov(self.close_obs[:, t0:t+1, :][0].T), axis=0)
            covariance = np.concatenate([high_cov, low_cov, close_cov], axis=0) # shape(3, 8, 8)
            
            portfolio = np.concatenate([self.high_obs, self.low_obs, self.close_obs], axis=0) # shape(3, 1042, 8)
            portfolio = portfolio[:, t0:t+1, :] # (3, 60, 8)
            portfolio = self._kalman(np.transpose(portfolio, [0, 2, 1]))
            
            observation = {'portfolio':portfolio,
                           'action': self.weights,
                           'covariance':covariance}
        
        return observation