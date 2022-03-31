import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import itertools

from collections import deque
from model import Actor, Critic
# from utils import  OUNoise


class TD3:
    def __init__(self, **kwargs):
        
        for key, value in kwargs.item():
            setattr(self, key, value)
        
        self.s_dim = self.env.observation_space.shape[0]
        self.a_dim = self.env.action_space.shape[0]
        
        # initialize network
        self.actor = Actor(self.s_dim, 256, self.a_dim).to(self.device)
        self.actor_target = Actor(self.s_dim, 256, self.a_dim).to(self.device)
        self.critic_1 = Critic(self.s_dim+self.a_dim, 256, self.a_dim).to(self.device)
        self.critic_1_target = Critic(self.s_dim+self.a_dim, 256, self.a_dim).to(self.device)
        self.critic_2 = Critic(self.s_dim+self.a_dim, 256, self.a_dim).to(self.device)
        self.critic_2_target = Critic(self.s_dim+self.a_dim, 256, self.a_dim).to(self.device)
        
        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=self.ACTOR_LR)
        self.critic_1_optimizer = optim.RMSprop(self.critic_1.parameters(), lr=self.CRITIC_LR)
        self.critic_2_optimizer = optim.RMSprop(self.critic_2.parameters(), lr=self.CRITIC_LR)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        self.replay_buffer = deque(maxlen=self.MEMORY_SIZE)
        self.critic_update_iteration = 0
        self.actor_update_iteration = 0
        self.num_training = 0
    
    def _soft_update(self, net_target, net):
        for target_param, param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data*(1.0-self.TAU)+param.data*self.TAU)
    
    def _choose_action(self, s0):
        s0 = torch.tensor(s0, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.actor(s0).squeeze(0).cpu().detach().numpy()
    
    def _update_memory(self, *transitions):
        self.replay_memory.append(transitions)
    
    def _update_Q(self, s0, a0, s1, r1):
        noise = torch.ones_like(a0).data.normal_(0, self.POLICY_NOISE).to(self.device)
        noise = noise.clamp(-self.NOISE_CLIP, self.NOISE_CLIP)
        a1 = self.actor_target(s1) + noise
        a1 = a1.clamp(-self.MAX_ACTION, self.MAX_ACTION)
        
        target_Q1 = self.critic_1(s0, a1)
        target_Q2 = self.critic_1(s0, a1)
        target_Q = r1 + self.GAMMA * min(target_Q1, target_Q2).detach()
        
        # Optimize Critic 1
        current_Q1 = self.critic_1(s0, a0)
        Q1_loss = F.smooth_l1_loss(current_Q1, target_Q)
        self.critic_1_optimizer.zero_grad()
        Q1_loss.backward()
        self.critic_1_optimizer.step()
        
        # Optimize Critic 2
        current_Q2 = self.critic_2(s0, a0)
        Q2_loss = F.smooth_l1_loss(current_Q2, target_Q)
        self.critic_2_optimizer.zero_grad()
        Q2_loss.backward()
        self.critic_2_optimizer.step()
    
    def _update_policy(self, s0):
        actor_loss = -torch.mean((self.critic_1(s0, self.actor(s0)) + self.critic_2(s0, self.actor(s0)))/2)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
    def _optimize(self, step):
        
        if len(self.replay_memory) < self.BATCH_SIZE:
            return
        
        samples = random.sample(self.replay_memory, self.BATCH_SIZE)
        
        s0, a0, r1, s1 = zip(*samples)
                
        s0 = torch.tensor(s0, dtype=torch.float, device=self.device)
        a0 = torch.tensor(a0, dtype=torch.float, device=self.device)
        r1 = torch.tensor(r1, dtype=torch.float, device=self.device).view(self.BATCH_SIZE,-1)
        s1 = torch.tensor(s1, dtype=torch.float, device=self.device)
        
        self._update_Q(s0, a0, s1, r1)
        
        if step % self.POLICY_DELAY == 0:
            self._update_policy(s0)
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_1_target, self.critic_1)
            self._soft_update(self.critic_2_target, self.critic_2)
    
    def train(self):
        
        for episode in range(self.EPISODES):
            s0 = self.env.reset()
            episode_reward = 0
            done = False
            for step in itertools.count():
                a0 = self._choose_action(s0)
                a0 += np.random.normal(0, self.EXPLORATION_NOISE, size=self.env.action_space.shape[0])
                a0 = a0.clip(self.env.action_space.low, self.env.action_space.high)
                s1, r1, done, _ = self.env.step(a0)
                episode_reward += r1
                if done:
                    break
                self._update_memory(s0, a0, s1, r1)
                self._optimize(step)
                s0 = s1
            
            print(episode, ': ', episode_reward)