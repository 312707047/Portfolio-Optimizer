from abc import ABC, abstractmethod
import numpy as np
import torch


class BaseReplayBuffer(ABC):
    
    def __init__(self, maxlen):
        
        self.memory = None
        self.maxlen = maxlen
        self.next_idx = 0
        self.is_full = False
        self.nowlen = self.maxlen if self.is_full else self.next_idx
    
    def __len__(self):
        return self.nowlen
    
    @abstractmethod
    def _sample(self, batch_size, **kwargs):
        pass
    
    def _update(self, inputs):
        if not self.is_full:
            self.memory[self.next_idx] = inputs
            self.next_idx += 1
            
            if self.next_idx >= self.maxlen:
                self.is_full = True
        else:
            self.memory = np.concatenate([self.memory[1:], np.expand_dims(inputs, axis=0)])


class PVM(BaseReplayBuffer):
    def __init__(self, maxlen, action_dim):
        super(BaseReplayBuffer).__init__(self, maxlen)
        
        self.memory = np.empty((maxlen, action_dim[0]), dtype=np.float32)
    
    def _sample(self, batch_size, index, device):
        ''' Sample action and previous action'''
        memory = np.stack((self.memory[index-1:index-1+batch_size],
                           self.memory[index:index+batch_size])) #shape(2, action_dim)
        
        memory = torch.tensor(memory, dtype=torch.float32, device=device)
        return memory


class OVM(BaseReplayBuffer):
    def __init__(self, maxlen, obs_dim):
        super(BaseReplayBuffer).__init__(self, maxlen)
        
        self.memory = np.empty((maxlen, obs_dim[0], obs_dim[1], obs_dim[2]), dtype=np.float32)
        
    def _sample(self, batch_size, index, device):
        memory = np.stack((self.memory[index:index+batch_size],
                           self.memory[index+1:index+1+batch_size]))
        memory = torch.tensor(memory, dtype=np.float32, device=device)
        return memory


class RewardMaskMemory(BaseReplayBuffer):
    def __init__(self, maxlen, gamma):
        super(BaseReplayBuffer).__init__(self, maxlen)
        
        self.memory = np.empty((maxlen, 2), dtype=np.float32)
        self.gamma = gamma
    
    def _sample(self, batch_size, index, device):
        memory = self.memory[index:index+batch_size]
        memory = torch.tensor(memory, dtype=np.float32, device=device)
        return memory
    
    def _update(self, reward, done):
        if not self.is_full:
            self.memory[self.next_idx, 0] = reward
            self.memory[self.next_idx, 1] = (1-done)*self.gamma
            self.next_idx += 1
            
            if self.next_idx >= self.maxlen:
                self.is_full = True
        else:
            self.memory = np.concatenate([self.memory[1:], np.expand_dims(input, axis=0)])


class ReplayBuffer:
    def __init__(self, maxlen:int, action_dim:tuple, obs_dim:tuple, gamma:float, device):
        
        # Initialize memory
        self.action_memory = PVM(maxlen=maxlen, action_dim=action_dim)
        self.observation_memory = OVM(maxlen=maxlen, obs_dim=obs_dim)
        self.reward_mask_memory = RewardMaskMemory(maxlen=maxlen, gamma=gamma)
        
        self.device = device
    
    def sample(self, batch_size):
        
        # Sample continuous data from the replay memory, not just random
        # start from 1 and minus 1 is to prevent invalid index.
        index = np.random.randint(1, self.action_memory.nowlen-batch_size-1)
        
        action = self.action_memory._sample(batch_size, index, self.device)
        observation = self.observation_memory._sample(batch_size, index, self.device)
        reward_mask = self.reward_mask_memory._sample(batch_size, index, self.device)
        
        # return state, action with previous one, reward, done, next_state
        return observation[0], action, reward_mask[0], observation[1], reward_mask[1]
    
    def update(self, state, action, reward, done):
        
        # `next_state`` is the next state, so no need to record it.
        self.action_memory._update(action)
        self.observation_memory._update(state)
        self.reward_mask_memory._update(reward, done)