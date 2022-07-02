from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self):
        
        self.actor = None
        self.actor_target = None
        self.actor_optimizer = None
        
        self.critic = None
        self.critic_target = None
        self.critic_optimizer = None
    
    @abstractmethod
    def update_memory(self, **kwargs):
        return NotImplemented
    
    @abstractmethod
    def optimize(self, **kwargs):
        return NotImplemented
    
    @abstractmethod
    def train(self, **kwargs):
        return NotImplemented
    
    @abstractmethod
    def test(self, **kwargs):
        return NotImplemented
        
    