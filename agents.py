import numpy as np
import logging
import random
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque
from parameter import DDPGparam, TD3param
from model import QNetwork, PolicyNetwork
from utils import  LinearAnneal, StateProcessor

class DDPG(DDPGparam):
    def __init__(self, env, device):
        self.env = env
        self.device = device
        
        # initialize model and optimizer
        self.policy_online = PolicyNetwork()
        self.policy_target = PolicyNetwork()
        self.Q_online = QNetwork()
        self.Q_target = QNetwork()
        self.policy_target.load_state_dict(self.policy_online.state_dict())
        self.Q_target.load_state_dict(self.Q_online.state_dict())
        
        self.policy_optimizer = optim.RMSprop(self.policy_online.parameters(), lr=self.LR)
        self.Q_optimizer = optim.RMSprop(self.Q_online.parameters(), lr=self.LR)
        
        # initialize memory
        self.replay_memory = deque(maxlen=self.MEMORY_SIZE)

        # initialize logging
        self._init_log()
    
    def soft_update(self, net_target, net):
        for target_param, param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data*(1.0-self.TAU)+param.data*self.TAU)
    
    def _choose_action(self, s0):
        s0 = torch.tensor(s0, dtype=torch.float32).unsqueeze(0)
        return self.policy_online(s0).squeeze(0).detach().numpy()
    
    def _update_memory(self, *transitions):
        self.replay_memory.append(transitions)
    
    def _update_policy(self, s0):
        loss = -torch.mean( self.Q_online(s0, self.policy_online(s0)) )
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
    
    def _update_Q(self, s0, a0, s1, r1):
        a1 = self.policy_target(s1).detach()
        y_true = r1 + self.GAMMA * self.Q_target(s1, a1).detach()
        
        y_pred = self.Q_online(s0, a0)
        
        loss_fn = nn.MSELoss()
        loss = loss_fn(y_pred, y_true)
        self.Q_optimizer.zero_grad()
        loss.backward()
        self.Q_optimizer.step()
    
    def _optimize(self):
        if len(self.replay_memory) < self.BATCH_SIZE:
            return
        
        samples = random.sample(self.buffer, self.batch_size)
        
        s0, a0, r1, s1 = zip(*samples)
                
        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size,-1)
        s1 = torch.tensor(s1, dtype=torch.float)
        
        self._update_policy(s0)
        self._update_Q(s0, a0, s1, r1)
        self.soft_update(self.Q_target, self.Q_online)
        self.soft_update(self.policy_target, self.policy_online)
    
    def train(self):
        for episode in range(100):
            s0 = self.env.reset()
            episode_reward = 0

            while not done:
                self.env.render()
                a0 = self._choose_action(s0)
                s1, r1, done, _ = self.env.step(a0)
                self._update_memory(s0, a0, r1, s1)

                episode_reward += r1
                s0 = s1

                self._optimize()

            print(episode, ': ', episode_reward)


class TD3(TD3param):
    def __init__(self, env, device):
        self.env = env
        self.device = device
        
        # initialize model and optimizer
        self.policy_online = PolicyNetwork()
        self.policy_target = PolicyNetwork()
        self.Q_online = QNetwork()
        self.Q_target = QNetwork()
        self.target_Q_online = QNetwork()
        self.target_Q_target = QNetwork()
        self.policy_target.load_state_dict(self.policy_online.state_dict())
        self.target_Q_target.load_state_dict(self.Q_target.state_dict())
        self.target_Q_online.load_state_dict(self.Q_online.load_state_dict())
        
        self.policy_optimizer = optim.RMSprop(self.policy_online.parameters(), lr=self.LR)
        self.Q_online_optimizer = optim.RMSprop(self.target_Q_online.parameters(), lr=self.LR)
        self.Q_target_optimizer = optim.RMSprop(self.target_Q_target.parameters(), lr=self.LR)
        
        # initialize memory
        self.replay_memory = deque(maxlen=self.MEMORY_SIZE)

        # initialize logging
        self._init_log()
        
        self.update_cnt = 0
    
    def soft_update(self, net_target, net):
        for target_param, param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data*(1.0-self.TAU)+param.data*self.TAU)
    
    def _choose_action(self, s0):
        s0 = torch.tensor(s0, dtype=torch.float32).unsqueeze(0)
        return self.policy_online(s0).squeeze(0).detach().numpy()
    
    def _update_memory(self, *transitions):
        self.replay_memory.append(transitions)
    
    def _update_policy(self, s0):
        loss = -torch.mean( self.Q_online(s0, self.policy_online(s0)) )
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
    
    def _update_Q(self, s0, a0, s1, r1):
        noise = torch.ones_like(a0).data.normal_(0, self.POLICY_NOISE).to(self.device)
        noise = noise.clamp(-self.NOISE_CLIP, self.NOISE_CLIP)
        a1 = (self.policy_target(s1) + noise).detach()
        
        # Compute Q value
        target_Q1 = self.target_Q_online(s1, a1)
        target_Q2 = self.target_Q_target(s1, a1)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = r1 + self.GAMMA * target_Q
        
        y_pred = self.Q_online(s0, a0)
        
        loss_fn = nn.MSELoss()
        loss = loss_fn(y_pred, y_true)
        self.Q_optimizer.zero_grad()
        loss.backward()
        self.Q_optimizer.step()
    
    def _optimize(self):
        if len(self.replay_memory) < self.BATCH_SIZE:
            return
        
        samples = random.sample(self.buffer, self.batch_size)
        
        s0, a0, r1, s1 = zip(*samples)
                
        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size,-1)
        s1 = torch.tensor(s1, dtype=torch.float)
        
        noise = torch.ones_like(a0).data.normal_(0, self.POLICY_NOISE).to(self.device)
        noise = noise.clamp(-self.NOISE_CLIP, self.NOISE_CLIP)
        a1 = (self.policy_online(s1) + noise)
        a1 = a1.clamp(-1, 1) # max action
        
        self._update_policy(s0)
        self._update_Q(s0, a0, s1, r1)
        self.soft_update(self.Q_target, self.Q_online)
        self.soft_update(self.policy_target, self.policy_online)
    
    def train(self):
        for episode in range(100):
            s0 = self.env.reset()
            episode_reward = 0

            while not done:
                self.env.render()
                a0 = self._choose_action(s0)
                s1, r1, done, _ = self.env.step(a0)
                self._update_memory(s0, a0, r1, s1)

                episode_reward += r1
                s0 = s1

                self._optimize()

            print(episode, ': ', episode_reward)