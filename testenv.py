import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym.envs.registration import register

def portfolio(returns, weights):
    weights = np.array(weights)
    rets = returns.mean() * 252
    covs = returns.cov() * 252
    P_ret = np.sum(rets * weights)
    P_vol = np.sqrt(np.dot(weights.T, np.dot(covs, weights)))
    P_sharpe = P_ret / P_vol
    return np.array([P_ret, P_vol, P_sharpe])

"""Markowitz"""

class CryptoEnvironment:
    
    def __init__(self, prices = 'C:/Users/Timmy/Desktop', capital = 1e5):       
        self.prices = prices  
        self.capital = capital  
        self.data = self.load_data()

    def load_data(self):
        data =  pd.read_csv(self.prices)
        try:
            data.index = data['Date']
            data = data.drop(columns = ['Date'])
        except:
            data.index = data['date']
            data = data.drop(columns = ['date'])            
        return data
    
    def preprocess_state(self, state):
        return state
    
    def get_state(self, t, lookback, is_cov_matrix = True, is_raw_time_series = False):
        
        assert lookback <= t
        
        decision_making_state = self.data.iloc[t-lookback:t]
        decision_making_state = decision_making_state.pct_change().dropna()

        if is_cov_matrix:
            x = decision_making_state.cov()
            return x
        else:
            if is_raw_time_series:
                decision_making_state = self.data.iloc[t-lookback:t]
            return self.preprocess_state(decision_making_state)

    def get_reward(self, action, action_t, reward_t, alpha = 0.01):
        
        def local_portfolio(returns, weights):
            weights = np.array(weights)
            rets = returns.mean() # * 252
            covs = returns.cov() # * 252
            P_ret = np.sum(rets * weights)
            P_vol = np.sqrt(np.dot(weights.T, np.dot(covs, weights)))
            P_sharpe = P_ret / P_vol
            return np.array([P_ret, P_vol, P_sharpe])

        data_period = self.data[action_t:reward_t]
        weights = action
        returns = data_period.pct_change().dropna()
      
        sharpe = local_portfolio(returns, weights)[-1]
        sharpe = np.array([sharpe] * len(self.data.columns))          
        rew = (data_period.values[-1] - data_period.values[0]) / data_period.values[0]
        
        return np.dot(returns, weights), rew
        


class ETFEnvironment:
    
    def __init__(self, volumes = 'C:/Users/Timmy/Desktop',
                       prices = 'C:/Users/Timmy/Desktop',
                       returns = 'C:/Users/Timmy/Desktop', 
                       capital = 1e6):
        
        self.returns = returns
        self.prices = prices
        self.volumes = volumes   
        self.capital = capital  
        
        self.data = self.load_data()

    def load_data(self):
        volumes = np.genfromtxt(self.volumes, delimiter=',')[2:, 1:]
        prices = np.genfromtxt(self.prices, delimiter=',')[2:, 1:]
        returns=pd.read_csv(self.returns, index_col=0)
        assets=np.array(returns.columns)
        dates=np.array(returns.index)
        returns=returns.as_matrix()
        return pd.DataFrame(prices, 
             columns = assets,
             index = dates
            )
    
    def preprocess_state(self, state):
        return state
    
    def get_state(self, t, lookback, is_cov_matrix = True, is_raw_time_series = False):
        
        assert lookback <= t
        
        decision_making_state = self.data.iloc[t-lookback:t]
        decision_making_state = decision_making_state.pct_change().dropna()

        if is_cov_matrix:
            x = decision_making_state.cov()
            return x
        else:
            if is_raw_time_series:
                decision_making_state = self.data.iloc[t-lookback:t]
            return self.preprocess_state(decision_making_state)

    def get_reward(self, action, action_t, reward_t):
        
        def local_portfolio(returns, weights):
            weights = np.array(weights)
            rets = returns.mean() # * 252
            covs = returns.cov() # * 252
            P_ret = np.sum(rets * weights)
            P_vol = np.sqrt(np.dot(weights.T, np.dot(covs, weights)))
            P_sharpe = P_ret / P_vol
            return np.array([P_ret, P_vol, P_sharpe])
        
        weights = action
        returns = self.data[action_t:reward_t].pct_change().dropna()
        
        rew = local_portfolio(returns, weights)[-1]
        rew = np.array([rew] * len(self.data.columns))
        
        return np.dot(returns, weights), rew

"""以下為一般環境"""

class TradeEnv():
    def __init__(self, path = 'C:/Users/Timmy/Desktop', window_length=60,
                 portfolio_value= 100000, trading_cost= 1/100,interest_rate= 0.33/100, train_size = 0.7):
        
        #path to numpy data
        self.path = path
        #load the whole data
        self.data = np.load(self.path)


        #parameters
        self.portfolio_value = portfolio_value
        self.window_length=window_length
        self.trading_cost = trading_cost
        self.interest_rate = interest_rate

        #number of stocks and features
        self.nb_stocks = self.data.shape[1]
        self.nb_features = self.data.shape[0]
        self.end_train = int((self.data.shape[2]-self.window_length)*train_size)
        
        #init state and index
        self.index = None
        self.state = None
        self.done = False

        #init seed
        self.seed()

    def return_pf(self):
        """
        return the value of the portfolio
        """
        return self.portfolio_value
        
    def readTensor(self,X,t):
        ## this is not the tensor of equation 18 
        ## need to batch normalize if you want this one 
        return X[ : , :, t-self.window_length:t ]
    
    def readUpdate(self, t):
        #return the return of each stock for the day t 
        return np.array([1+self.interest_rate]+self.data[-1,:,t].tolist())

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self, w_init, p_init, t=0 ):
        
        """ 
        This function restarts the environment with given initial weights and given value of portfolio

        """
        self.state= (self.readTensor(self.data, self.window_length) , w_init , p_init )
        self.index = self.window_length + t
        self.done = False
        
        return self.state, self.done

    def step(self, action):

        index = self.index
        #get Xt from data:
        data = self.readTensor(self.data, index)
        done = self.done
        
        #beginning of the day 
        state = self.state
        w_previous = state[1]
        pf_previous = state[2]
        
        #the update vector is the vector of the opening price of the day divided by the opening price of the previous day
        update_vector = self.readUpdate(index)

        #allocation choice 
        w_alloc = action
        pf_alloc = pf_previous
        
        #Compute transaction cost
        cost = pf_alloc * np.linalg.norm((w_alloc-w_previous),ord = 1)* self.trading_cost
        
        #convert weight vector into value vector 
        v_alloc = pf_alloc*w_alloc
        
        #pay transaction costs
        pf_trans = pf_alloc - cost
        v_trans = v_alloc - np.array([cost]+ [0]*self.nb_stocks)
        
        #####market prices evolution 
        #we go to the end of the day 
        
        #compute new value vector 
        v_evol = v_trans*update_vector

        
        #compute new portfolio value
        pf_evol = np.sum(v_evol)
        
        #compute weight vector 
        w_evol = v_evol/pf_evol
        
        
        #compute instanteanous reward
        reward = (pf_evol-pf_previous)/pf_previous
        
        #update index
        index = index+1
        
        #compute state
        
        state = (self.readTensor(self.data, index), w_evol, pf_evol)
        
        if index >= self.end_train:
            done = True
        
        self.state = state
        self.index = index
        self.done = done
        
        return state, reward, done
        
        
        
        
        
        
 