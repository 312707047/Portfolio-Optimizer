from abc import ABC, abstractmethod
from environment.env import TradingEnv

class BaseAgent(ABC):
    def __init__(self, env:TradingEnv, gamma:float, actor_lr, critic_lr,
                 batch_size:int, memory_size:int, episodes:int, device:int):
        
        self.env = env
        self.gamma = gamma
        self.device = device
        self.episodes = episodes
        self.batch_size = batch_size
        self.memory_size = memory_size
        
        self.actor = None
        self.actor_target = None
        self.actor_lr = actor_lr
        self.actor_optimizer = None
        
        self.critic = None
        self.critic_target = None
        self.critic_lr = critic_lr
        self.critic_optimizer = None
    
    @abstractmethod
    def optimize(self, **kwargs):
        pass
    
    @abstractmethod
    def train(self, **kwargs):
        pass
    
    @abstractmethod
    def test(self, **kwargs):
        pass
        
    