import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
from model import Actor, Critic
from utils import  OUNoise

class DDPG:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        s_dim = self.env.observation_space.shape[0]
        self.a_dim = self.env.action_space.shape[0]
        
        # initialize model and optimizer
        self.actor = Actor(s_dim, 256, self.a_dim).to(self.device)
        self.actor_target = Actor(s_dim, 256, self.a_dim).to(self.device)
        self.critic = Critic(s_dim+self.a_dim, 512, self.a_dim).to(self.device)
        self.critic_target = Critic(s_dim+self.a_dim, 512, self.a_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=self.ACTOR_LR)
        self.critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=self.CRITIC_LR)
        
        # initialize memory
        self.replay_memory = deque(maxlen=self.MEMORY_SIZE)

        # initialize logging
        # self._init_log()
    
    def _soft_update(self, net_target, net):
        for target_param, param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data*(1.0-self.TAU)+param.data*self.TAU)
    
    def _choose_action(self, s0):
        s0 = torch.tensor(s0, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.actor(s0).squeeze(0).cpu().detach().numpy()
    
    def _update_memory(self, *transitions):
        self.replay_memory.append(transitions)
    
    def _update_policy(self, s0):
        loss = -torch.mean( self.critic(s0, self.actor(s0)))
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
    
    def _update_Q(self, s0, a0, s1, r1):
        a1 = self.actor_target(s1).detach()
        y_true = r1 + self.GAMMA * self.critic_target(s1, a1).detach()
        
        y_pred = self.critic(s0, a0)
        
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(y_pred.to(self.device), y_true.to(self.device))
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
    
    def _optimize(self):
        if len(self.replay_memory) < self.BATCH_SIZE:
            return
        
        samples = random.sample(self.replay_memory, self.BATCH_SIZE)
        
        s0, a0, r1, s1 = zip(*samples)
                
        s0 = torch.tensor(s0, dtype=torch.float, device=self.device)
        a0 = torch.tensor(a0, dtype=torch.float, device=self.device)
        r1 = torch.tensor(r1, dtype=torch.float, device=self.device).view(self.BATCH_SIZE,-1)
        s1 = torch.tensor(s1, dtype=torch.float, device=self.device)
        
        self._update_policy(s0)
        self._update_Q(s0, a0, s1, r1)
        self._soft_update(self.critic_target, self.critic)
        self._soft_update(self.actor_target, self.actor)
    
    def train(self):
        ou_noise = OUNoise(self.env.action_space)
        for episode in range(self.EPISODES):
            s0 = self.env.reset()
            episode_reward = 0
            ou_noise.reset()
            
            for step in range(500):
                # self.env.render()
                a0 = self._choose_action(s0)
                a0 = ou_noise.get_action(a0, step)
                s1, r1, done, _ = self.env.step(a0)
                self._update_memory(s0, a0, r1, s1)

                episode_reward += r1
                s0 = s1

                self._optimize()

            print(episode, ': ', episode_reward)