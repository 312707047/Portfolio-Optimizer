from base.base_replaybuffer import BaseReplayBuffer
import numpy as np
import torch



class BaselineBuffer(BaseReplayBuffer):
    def __init__(self, maxlen, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim, maxlen)
        memo_dim = 1 + 1 + obs_dim + action_dim + obs_dim
        self.memories = np.empty((maxlen, memo_dim), dtype=np.float32)

        self.next_idx = 0
        self.is_full = False
        self.max_len = maxlen
        self.now_len = self.max_len if self.is_full else self.next_idx

        self.state_idx = 1 + 1 + obs_dim  # reward_dim==1, done_dim==1
        self.action_idx = self.state_idx + action_dim

    def update(self, memo_tuple):
        self.memories[self.next_idx] = np.hstack(memo_tuple)
        self.next_idx = self.next_idx + 1
        if self.next_idx >= self.max_len:
            self.is_full = True
            self.next_idx = 0
    
    def sample(self, batch_size, device):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # indices = rd.choice(self.memo_len, batch_size, replace=False)  # why perform worse?
        # indices = rd.choice(self.memo_len, batch_size, replace=True)  # why perform better?
        # same as:
        indices = np.random.randint(self.now_len, size=batch_size)

        memory = self.memories[indices]
        memory = torch.tensor(memory, device=device)

        '''convert array into torch.tensor'''
        tensors = (
            memory[:, 0:1],  # rewards
            memory[:, 1:2],  # masks, mark == (1-float(done)) * gamma
            memory[:, 2:self.state_idx],  # states
            memory[:, self.state_idx:self.action_idx],  # actions
            memory[:, self.action_idx:],  # next_states
        )
        return tensors
            
class ReplayBuffer(BaselineBuffer):
    
    def __init__(self, maxlen, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim, maxlen)
        self.obs_memory = np.empty((maxlen, obs_dim[0], obs_dim[1], obs_dim[2]), dtype=np.float32)
        self.act_memory = np.empty((maxlen, action_dim[0], action_dim[1], action_dim[2]), dtype=np.float32)
        self.other_memory = np.empty((maxlen, 2), dtype=np.float32)
        
        self.next_idx = 0
        self.is_full = False
        self.nowlen = self.maxlen if self.is_full else self.next_idx
    
    def update(self, state, action, reward, done):
        if not self.is_full:
            self.other_memory[self.next_idx] = np.stack((reward, done))
            self.obs_memory[self.next_idx] = state['observation']
            self.act_memory[self.next_idx] = action
            self.next_idx += 1
        
            # if the replay buffer is full
            if self.next_idx >= self.maxlen:
                self.is_full = True
                self.next_idx = 0
        
        else:
            self.other_memory = np.concatenate([self.other_memory[1:], np.expand_dims(np.stack((reward, done)), axis=0)])
            self.obs_memory = np.concatenate([self.obs_memory[1:], np.expand_dims(state['observation'], axis=0)])
            self.act_memory = np.concatenate([self.act_memory[1:], np.expand_dims(action, axis=0)])
    
    def sample(self, batch_size):
        
        # Sample continuous data from the replay memory, not just random
        index = np.random.randint(1, self.nowlen-batch_size-1)
        
        state = self.obs_memory[index:index+batch_size]
        action = self.act_memory[index:index+batch_size]
        pre_act = self.act_memory[index-1:index+batch_size-1]
        reward = self.other_memory[index:index+batch_size, 0:1]
        done = self.other_memory[index:index+batch_size, 1:2]
        next_state = self.obs_memory[index+1:index+batch_size+1]
        
        state = {'observation':state, 'action':pre_act}
        next_state = {'observation':next_state, 'action':action}
        
        return state, action, reward, next_state, done
    
    def __len__(self):
        return self.next_idx