import numpy as np

from abc import ABC, abstractmethod

class BaseReplayBuffer(ABC):
    
    def __init__(self, obs_dim, action_dim, maxlen):
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.maxlen = maxlen
    
    @abstractmethod
    def sample(self, batch_size):
        pass
    
    @abstractmethod
    def update(self, **kwargs):
        pass