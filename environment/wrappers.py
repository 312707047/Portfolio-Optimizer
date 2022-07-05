import gym
import numpy as np
import torch

from pykalman import KalmanFilter
from sklearn.preprocessing import MinMaxScaler
from .env import TradingEnv
# from scipy.special import softmax

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

EPS = 1e-8


class TransposeSpace(gym.ObservationWrapper):
    def __init__(self, env:TradingEnv):
        super().__init__(env)

        self.observation_space = gym.spaces.Dict({'observation': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3, env.tickers_num, env.rolling_window), dtype=np.float32),
                  'action': self.action_space})
        
    def observation(self, obs):
        data = obs['observation'].transpose(0, 2, 1)
        obs.update({'observation': data})
        
        return obs
        

class kalmanfilter(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        self.kf = KalmanFilter(transition_matrices = [1],
                               observation_matrices = [1],
                               initial_state_mean = 0,
                               initial_state_covariance = 1.5,
                               observation_covariance = 1.5,
                               transition_covariance = 1/30)
    
    def observation(self, obs:dict):
        data = obs['observation']
        for i in range(len(data)):
            for j in range(len(data[i])):
                a, _ = self.kf.filter(data[i][j])
                data[i][j] = np.squeeze(a)
        obs.update({'observation': data})
        
        return obs


class scaler(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, obs:dict):
        data = obs['observation']
        data = data.transpose(0, 2, 1)
        for i in range(data.shape[0]):
            scaler = MinMaxScaler()
            data[i, :, :] = scaler.fit_transform(data[i, :, :])
        
        data = data.transpose(0, 2, 1)
        obs.update({'observation': data})
        
        return obs
    

def softmax(w, t=1.0):
    """softmax implemented in numpy."""
    log_eps = np.log(EPS)
    w = np.clip(w, log_eps, -log_eps)  # avoid inf/nan
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist


class SoftmaxAction(gym.Wrapper):
    
    def step(self, action):
        
        action = softmax(action)
        action = np.concatenate([action[:3]*0.1, action[3:]*0.9])
        action = action / action.sum()
        # print(action)
        return self.env.step(action)


def env_wrapper(env):
    
    env = SoftmaxAction(env)
    env = TransposeSpace(env)
    
    # if mmscaler:
    #     env = scaler(env)
    # if kf:
    #     env = kalmanfilter(env)
    # if enc:
    #     env = Encoder(env)
    
    return env