import gym
import os
import numpy as np
import pandas as pd

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
        
        close_prices = pd.read_csv(os.path.join(data_path, 'Close.csv'), index_col=0, parse_dates=True)
        self.tickers = close_prices.columns.to_list()
        self.tickers_num = len(self.tickers)
        self.dates = close_prices.index.values[1:]
        self.dates_num = self.dates.shape[0]
        self.gain = np.hstack((np.ones((close_prices.shape[0]-1, 1)), close_prices.values[1:] / close_prices.values[:-1]))
        
        self.info = []
        self.step_number = 0
        
        # Observation and action space for the agent
        
        self.action_space = gym.spaces.Box(
            0, 1, shape=(len(self.data_num)+1, ), dtype=np.float32)  # include cash
        
        close_obs = np.expand_dims(close_prices.T, 0) # shape (1, 8, 815)
        
        if observation_features == 'Close':
            self.observation = close_obs
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.tickers_num, rolling_window), dtype=np.float32)
        
        elif observation_features == 'Three':
            
            high_prices = pd.read_csv(os.path.join(data_path, 'High.csv'), index_col=0, parse_dates=True)
            low_prices = pd.read_csv(os.path.join(data_path, 'Low.csv'), index_col=0, parse_dates=True)
            
            high_obs = np.expand_dims(high_prices.T, 0)
            low_obs = np.expand_dims(low_prices.T, 0)
            
            self.observation = np.concatenate([high_obs, low_obs, close_obs], axis=0) #shape(3, 8, 815)
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3, self.tickers_num, rolling_window), dtype=np.float32)

        elif observation_features == 'All':
            spaces = {
                'portfolio': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.tickers_num, rolling_window, 3), dtype=np.float32),
                'covariance': gym.spaces.Box(low=-1.0, high=1.0, shape=(3, self.tickers_num, self.tickers_num), dtype=np.float32)
            }
            
            self.observation_space = gym.spaces.Dict(spaces)
            ########## finish observation space ##########
            
            # TO DO:
            # 1. Calculate covariance matrix of every step
            # 2. Dealing with observation Dict
            
            ##############################################
        
        self.start_date_index = start_date_index
        self.steps = steps
        self.reset()
    
    def step(self, action):
        
        self.step_number += 1
        
        w1 = np.clip(action, a_min=0, a_max=1)
        w1 = np.insert(w1, 0, np.clip(1 - w1.sum(), a_min=0, a_max=1))
        w1 = w1 / w1.sum()
        
        # calculate the reward
        t = self.start_date_index + self.step_number
        y1 = self.gain[t]
        w0 = self.weights
        p0 = self.portfolio_value
        dw1 = (y1 * w0) / (np.dot(y1, w0)+EPS)
        mu1 = self.commission * (np.abs(dw1 - w1)).sum()
        p1 = p0 * (1 - mu1) * np.dot(y1, w1)
        p1 = np.clip(p1, 0, np.inf)
        rho1 = p1 / p0 - 1
        reward = np.log((p1+EPS)/(p0+EPS))
        
        #################### TO DO ####################
        #
        # Reward Function:
        #   1. Calculate return of Markowitz
        #   2. Calculate return of same-weighted portfolio
        #   3. Calculate Sharpe, DD, MDD, etc
        #
        # Reward Shaping:
        #   1. Avoid digicurrencies have more than 10% weight
        #   2. Avoid single asset has more than 65% weight
        #
        
        port_returns = []
        port_volatility = []
        sharpe_ratio = []
        stock_weights = []
        
        num_portfolios = 50000   #這裡是用MC模擬50000次，實際上讓agent自己去跟環境互動即可

        np.random.seed(100)

        for single_portfolio in range(num_portfolios):
            weights = np.random.random(tickers_num)
            weights /= np.sum(weights)
            returns = np.dot(weights, reward.mean() * 250)   #交易天數250天
            volatility = np.sqrt(np.dot(weights.T, np.dot(reward.cov() * 250, weights)))
            sharpe = returns / volatility
            sharpe_ratio.append(sharpe)
            port_returns.append(returns)
            port_volatility.append(volatility)
            stock_weights.append(weights)

        portfolio = {'Returns': port_returns,          #return
                     'Volatility': port_volatility,
                     'Sharpe Ratio': sharpe_ratio}     #Sharpe Ratio
        
        for counter,symbol in enumerate(tickers):
            portfolio[symbol + 'Weight'] = [Weight[counter] for Weight in stock_weights]

        df = pd.DataFrame(portfolio)
        column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock + 'Weight' for stock in tickers]
        df = df[column_order]
        
        min_volatility = df['Volatility'].min()
        max_sharpe = df['Sharpe Ratio'].max()
        sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
        min_variance_port = df.loc[df['Volatility'] == min_volatility]    #return of Markowitz that minimizes the risk
        
        def calc_MDD(networth):    #calculate DD and MDD
            series = pd.Series(networth, name="nw").to_frame()

            max_peaks_idx = series.nw.expanding(min_periods=1).apply(lambda x: x.argmax()).fillna(0).astype(int)
            series['max_peaks_idx'] = pd.Series(max_peaks_idx).to_frame()

            nw_peaks = pd.Series(series.nw.iloc[max_peaks_idx.values].values, index=series.nw.index)

            series['dd'] = ((series.nw - nw_peaks)/nw_peaks)
            series['mdd'] = series.groupby('max_peaks_idx').dd.apply(lambda x: x.expanding(min_periods=1).apply(lambda y: y.min())).fillna(0)

            return series
        
        calc_MDD(df)
        ###############################################
        
        # save weights and portfolio value for next iteration
        self.weights = w1
        self.portfolio_value = p1
        
        # observe the next state
        t0 = t - self.rolling_window + 1
        observation = self.observation[:, :, t0:t+1]
        
        # info
        r = y1.mean()
        if self.step_number == 1:
            market_value = r
        else:
            market_value = self.info_list[-1]["market_value"] * r 
        info = {"reward": reward, "log_return": reward, "portfolio_value": p1, "return": r, "rate_of_return": rho1,
                "weights_mean": w1.mean(), "weights_std": w1.std(), "cost": mu1, 'date': self.dates[t],
                'steps': self.step_number, "market_value": market_value}
        self.info.append(info)
        
        # ckeck done
        done = False
        if (self.step_number >= self.steps) or (p1 <= 0):
            done = True
        
        return observation, reward, done, info
    
    def reset(self):
        
        self.info = []
        self.weights = np.insert(np.zeros(self.tickers_num+1), 0, 1.0)
        self.portfolio_value = 1.0
        self.step_number = 0
        
        self.steps = min(self.steps, self.dates_num - self.rolling_window - 1)
        
        if self.start_date_index is None:
            self.start_date_index = np.random.random_integers(self.rolling_window-1,
                                                              self.dates_num-self.steps-1)
        
        else:
            self.start_date_index = np.clip(self.start_date_index,
                                            a_min=self.rolling_window-1,
                                            a_max=self.dates_num-self.steps-1)
        
        t = self.start_date_index + self.step_number
        t0 = t - self.rolling_window + 1
        
        ################## TO DO ##################
        
        # Observation for 'All' mode
        
        ###########################################
        if self.observation_features == ('Close' or 'Three'):
            observation = self.observation[:, :, t0:t+1] # need to deal with the dict space
        
        return observation