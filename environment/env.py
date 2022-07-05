import gym
import os
import numpy as np
import pandas as pd

from data.load_data import RawDataLoader

EPS = 1e-8

class TradingEnv(gym.Env):
    def __init__(self, data_path,
                 rolling_window=30,
                 commission=0.01,
                 time_cost=0.0,
                 steps=200,
                 augment=0.00,
                 start_date_index=None):
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
        self.time_cost = time_cost
        self.data_path = data_path
        self.augment = augment
        
        self.tickers = [i[:-17] for i in os.listdir(self.data_path)]
        raw_data_loader = RawDataLoader(self.tickers)
        data = raw_data_loader.load_data(self.data_path)
        
        # read data
        self.close_prices = data['Close']
        self.open_prices = data['Open']
        self.high_prices = data['High']
        self.low_prices = data['Low']
        
        self.close_obs = np.expand_dims(self.close_prices, 0) #shape(1, 65321, 10)
        self.open_obs = np.expand_dims(self.open_prices, 0)
        self.high_obs = np.expand_dims(self.high_prices, 0)
        self.low_obs = np.expand_dims(self.low_prices, 0)
            
        self.tickers_num = len(self.tickers)
        self.dates = self.close_prices.index.values[1:]
        self.dates_num = self.dates.shape[0]
        
        # add cash to the gain
        self.gain = np.hstack((np.ones((self.close_prices.shape[0]-1, 1)), self.close_prices.values[1:] / self.close_prices.values[:-1]))
        self.gain = self.gain[1:] / self.gain[:-1]
        
        self.info = []
        self.step_number = 0
        
        # Observation space and action space
        self.action_space = gym.spaces.Box(
            0, 1, shape=(self.tickers_num+1,), dtype=np.float32)
            
        spaces = {'observation': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4, rolling_window, self.tickers_num), dtype=np.float32),
                  'action': self.action_space}
            
        self.observation_space = gym.spaces.Dict(spaces)
        self.start_date_index = start_date_index
        self.steps = steps
        self.reset()

    def step(self, action):
        
        self.step_number += 1
        
        # if cash is needed ### add this to neural network later
        # w1 = np.clip(action, a_min=0, a_max=1)
        # w1 = np.insert(w1, 0, np.clip(1 - w1.sum(), a_min=0, a_max=1))
        w1 = action
        
        # 1. Calculate agent reward
        t = self.start_date_index + self.step_number
        y1 = self.gain[t]
        
        w0 = self.weights
        p0 = self.portfolio_value
        dw1 = (y1 * w0) / (np.dot(y1, w0)+EPS)
        mu1 = self.commission * (np.abs(dw1 - w1)).sum()
        p1 = p0 * (1 - mu1) * np.dot(y1, w1)
        p1_augmented = p1 * (1 - self.time_cost) # make reward depreciate by time
        p1 = np.clip(p1, 0, np.inf)
        rho1 = p1 / p0 - 1
        agent_return = np.log((p1+EPS)/(p0+EPS))
        
        # calculate reward and scale reward between 1 and -1, and do reward shaping to avoid big weight
        reward = np.log((p1_augmented+EPS)/(p0+EPS)) #- (np.max(w1)*0.3)
        # reward = (agent_return - same_weighted_return) * 1000 #- 0.3 * max(w1)
        
        # observe the next state
        t0 = t - self.rolling_window + 1
        
        obs = self.portfolio[:, t0:t+1, :] # (3, 60, 8)
        obs /= obs[0, -1]
        
        observation = {'observation': obs, 'action': self.weights}
        
        # save weights and portfolio value for next iteration
        self.weights = w1
        self.portfolio_value = p1
        # self.same_weighted_portfolio_value = s_p1
        
        # 3. Calculate MDD
        # self.portfolio_value_list.append(p1)
        # DD = min(self.portfolio_value_list) / max(self.portfolio_value_list) - 1
        
        # 4. Check limitation and done
        done = False
        if self.step_number >= self.steps:
            done = True
        
        # Reward shaping: MDD
        # try:
        #     if min(self.DD) > DD:
        #         reward -= 0.2
        # except ValueError:
        #     pass
        
        # if DD < 0:
        #     self.DD.append(DD)
        
        if p1 <= 0.5:
            reward -=1
            done = True
        
        # info
        r = y1.mean()
        if self.step_number == 1:
            market_value = r
        else:
            market_value = self.info_list[-1]["market_value"] * r
 
        info = {"log_return": agent_return, "portfolio_value": round(p1, 5), "return": round(r, 3), "weights": np.around(w1, decimals=3),
                "weights_std": round(w1.std(), 3), "cost": round(mu1, 5), 'date': np.datetime_as_string(self.dates[t])[:10],
                'steps': self.step_number, "market_value": round(market_value, 3)}
        self.info_list.append(info)

        return observation, reward, done, info
    
    def reset(self):
        
        self.info_list = []
        self.weights = np.insert(np.zeros(self.tickers_num), 0, 1.0)
        self.portfolio_value = 1.0
        self.step_number = 0
        
        # Calculate MDD
        self.portfolio_value_list = [self.portfolio_value]
        self.DD = []
        
        self.steps = min(self.steps, self.dates_num - self.rolling_window - 1)
        
        if self.start_date_index is None:
            self.start_date_index = np.random.randint(self.rolling_window-1,
                                                      self.dates_num-self.steps-1)
        
        else:
            self.start_date_index = np.clip(self.start_date_index, a_min=self.rolling_window-1, a_max=self.dates_num-self.steps-1)
        
        t = self.start_date_index + self.step_number
        t0 = t - self.rolling_window + 1
        
        self.portfolio = np.concatenate([self.open_obs, self.high_obs, self.low_obs, self.close_obs], axis=0) # shape(4, 1042, 9)
        
        # add noise to the data to prevent overfitting
        self.portfolio += np.random.normal(loc=0, scale=self.augment, size=self.portfolio.shape)
        
        obs = self.portfolio[:, t0:t+1, :] # (3, 60, 8)
        obs /= obs[0, -1]

        observation = {'observation': obs, 'action': self.weights}
            
        return observation