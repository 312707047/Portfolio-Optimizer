import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import calculate_cov, output_recorder


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
    def __init__(self, device):
        super(TD3_Actor, self).__init__()
        self.device = device

        self.conv_port = nn.Conv2d(3, 2, (1, 3))
        # self.conv_cov = nn.Conv2d(3, 1, (1, 1))
        self.conv_mix = nn.Conv2d(2, 20, (1, 58))
        self.conv_out = nn.Conv2d(21, 1, 1)
    
    def forward(self, observation):
        
        port = torch.tensor(observation['observation'], dtype=torch.float32, device=self.device)
        port = port.view(-1, 3, 8, 60)
        # cov = calculate_cov(port).to(self.device)
        action = torch.tensor(observation['action'], dtype=torch.float32, device=self.device)
        action = action.view((-1, 1, 8, 1))
            
        port = F.leaky_relu(self.conv_port(port))
        # cov = F.leaky_relu(self.conv_cov(cov))
        # m = torch.concat([port, cov], dim=1)
        m = F.leaky_relu(self.conv_mix(port)) # shape(1, 20, 8, 1)
        all = torch.concat([m, action], dim=1) # shape(1, 21, 8, 1)
        all = self.conv_out(all)
        # output_recorder(self.model_type, port, 'conv_port')
        # output_recorder(self.model_type, cov, 'conv_cov')
        # output_recorder(self.model_type, m, 'conv_mix')
        # output_recorder(self.model_type, all, 'conv_out')
        
        return all.squeeze()
        

class TD3_Critic(nn.Module):
    def __init__(self, device):
        super(TD3_Critic, self).__init__()
        self.device = device
    
        # Q1
        self.conv_port1 = nn.Conv2d(3, 2, (1, 3))
        # self.conv_cov1 = nn.Conv2d(3, 1, (1, 1))
        self.conv_mix1 = nn.Conv2d(2, 20, (1, 58))
        self.conv_out1 = nn.Conv2d(22, 1, (8, 1))
        # self.linear1 = nn.Linear(8, 1)
        
        # Q2
        self.conv_port2 = nn.Conv2d(3, 2, (1, 3))
        # self.conv_cov2 = nn.Conv2d(3, 1, (1, 1))
        self.conv_mix2 = nn.Conv2d(2, 20, (1, 58))
        self.conv_out2 = nn.Conv2d(22, 1, (8, 1))
        # self.linear2 = nn.Linear(8, 1)
    
    def forward(self, observation, act):
        # port = torch.tensor(list(map(lambda x: x['portfolio'], observation)), dtype=torch.float32, device=self.device)
        # action = torch.tensor(list(map(lambda x: x['action'], observation)), dtype=torch.float32, device=self.device)
        
        port = torch.tensor(observation['observation'], dtype=torch.float32, device=self.device)
        port = port.view(-1, 3, 8, 60)
        # cov = calculate_cov(port).to(self.device)
        action = torch.tensor(observation['action'], dtype=torch.float32, device=self.device)
        action = action.view((-1, 1, 8, 1))
        act = act.view((-1, 1, 8, 1))
    
        # Q1
        q1_port = F.leaky_relu(self.conv_port1(port))
        # q1_cov = F.leaky_relu(self.conv_cov1(cov))
        # q1_m = torch.concat([q1_port, q1_cov], dim=1)
        q1_m = F.leaky_relu(self.conv_mix1(q1_port)) # shape(1, 20, 8, 1)
        q1_all = torch.concat([q1_m, action, act], dim=1) # shape(1, 22, 8, 1)
        q1_all = self.conv_out1(q1_all).squeeze()
        # q1 = self.linear1(q1_all)
        
        # Q2
        q2_port = F.leaky_relu(self.conv_port2(port))
        # q2_cov = torch.relu(self.conv_cov2(cov))
        # q2_m = torch.concat([q2_port, q2_cov], dim=1)
        q2_m = F.leaky_relu(self.conv_mix2(q2_port)) # shape(1, 20, 8, 1)
        q2_all = torch.concat([q2_m, action, act], dim=1) # shape(1, 22, 8, 1)
        q2_all = self.conv_out2(q2_all).squeeze()
        # q1 = self.linear1(q1_all)
            
        return q1_all, q2_all