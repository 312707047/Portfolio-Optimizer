import numpy as np
import logging
import random
import itertools

from collections import deque
from parameter import DDPGparam
from utils import  LinearAnneal

class DDPG(DDPGparam):
    def __init__(self, env):
        self.env = env
        
        # initialize model
        self._init_model()
        
        # initialize memory
        self.replay_memory = deque(maxlen=self.MEMORY_SIZE)
        
        # greedy strategy
        self.epsilon = LinearAnneal(self.EPSILON, self.MIN_EPSILON, self.EPISODES)
        
        # initialize logging
        self._init_log()
    
    def _init_model(self):
        pass
    
    def _update_model(self):
        pass
    
    def _update_memory(self, transitions):
        self.replay_memory.append(transitions)
    
    def _choose_action(self):