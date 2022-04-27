import torch
import torch.nn as nn
from agent.latent_space.LFSS import PretrainedAEModel


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, state):
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        
        return x


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        
        return x


class TD3_Actor(nn.Module):
    def __init__(self, device) -> None:
        super(TD3_Actor, self).__init__()
        self.conv_port = nn.Conv2d(3, 2, (1, 23))
        self.conv_cov = nn.Conv2d(3, 1, (1, 1))
        self.conv_mix = nn.Conv2d(3, 20, (1, 8))
        self.conv_out = nn.Conv2d(21, 1, 1)
        self.device = device
        self.ae = PretrainedAEModel(device=device)
        
    
    def forward(self, observation):
        ### TypeError: tuple indices must be integers or slices, not str
        port = torch.tensor(observation['portfolio'], dtype=torch.float32, device=self.device)
        cov = torch.tensor(observation['covariance'], dtype=torch.float32, device=self.device)
        action = torch.tensor(observation['action'], dtype=torch.float32, device=self.device)
        
        port = torch.relu(self.conv_port(self.ae.predict(port)))
        cov = torch.relu(self.conv_cov(cov))
        m = torch.concat([port, cov], dim=0) # shape(3, 8, 8)
        m = torch.relu(self.conv_mix(m)) # shape(20, 8, 1)
        action = torch.unsqueeze(torch.unsqueeze(action, dim=1), dim=0) # shape(1, 8, 1)
        all = torch.concat([m, action], dim=0) # shape(21, 8, 1)
        all = torch.softmax(self.conv_out(all), dim=1)
        
        return all.squeeze()
        

class TD3_Critic(nn.Module):
    def __init__(self, device):
        super(TD3_Critic, self).__init__()
        self.ae = PretrainedAEModel(device=device)
        self.device = device
        
        # Q1
        self.conv_port1 = nn.Conv2d(3, 2, (1, 23))
        self.conv_cov1 = nn.Conv2d(3, 1, (1, 1))
        self.conv_mix1 = nn.Conv2d(3, 20, (1, 8))
        self.conv_out1 = nn.Conv2d(21, 1, 1)
        self.linear1 = nn.Linear(16, 1)
        
        # Q2
        self.conv_port2 = nn.Conv2d(3, 2, (1, 23))
        self.conv_cov2 = nn.Conv2d(3, 1, (1, 1))
        self.conv_mix2 = nn.Conv2d(3, 20, (1, 8))
        self.conv_out2 = nn.Conv2d(21, 1, 1)
        self.linear2 = nn.Linear(16, 1)
    
    def forward(self, observation, act):
        port = torch.tensor(observation['portfolio'], dtype=torch.float32, device=self.device)
        cov = torch.tensor(observation['covariance'], dtype=torch.float32, device=self.device)
        action = torch.tensor(observation['action'], dtype=torch.float32, device=self.device)
        
        # Q1
        q1_port = torch.relu(self.conv_port1(self.ae.predict(port)))
        q1_cov = torch.relu(self.conv_cov1(cov))
        q1_m = torch.concat([q1_port, q1_cov], dim=0) # shape(3, 8, 8)
        q1_m = torch.relu(self.conv_mix1(q1_m)) # shape(20, 8, 1)
        q1_action = torch.unsqueeze(torch.unsqueeze(action, dim=1), dim=0) # shape(1, 8, 1)
        q1_all = torch.concat([q1_m, q1_action], dim=0) # shape(21, 8, 1)
        q1_all = torch.softmax(self.conv_out1(q1_all), dim=1).squeeze()
        q1 = torch.concat([q1_all, act])
        q1 = self.linear1(q1)
        
        
        # Q2
        q2_port = torch.relu(self.conv_port2(self.ae.predict(port).to(self.device)))
        q2_cov = torch.relu(self.conv_cov2(cov))
        q2_m = torch.concat([q2_port, q2_cov], dim=0) # shape(3, 8, 8)
        q2_m = torch.relu(self.conv_mix2(q2_m)) # shape(20, 8, 1)
        q2_action = torch.unsqueeze(torch.unsqueeze(action, dim=1), dim=0) # shape(1, 8, 1)
        q2_all = torch.concat([q2_m, q2_action], dim=0) # shape(21, 8, 1)
        q2_all = torch.softmax(self.conv_out2(q2_all), dim=1).squeeze()
        q2 = torch.concat([q2_all, act])
        q2 = self.linear1(q2)
        
        return q1, q2