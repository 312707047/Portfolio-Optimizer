import torch
import torch.nn as nn
from agent.latent_space.LFSS import PretrainedAEModel
from agent.latent_space.Denoise import Autoencoder


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
        self.ae = Autoencoder().to(self.device)
        self.ae.load_state_dict(torch.load('agent/latent_space/Autoencoder.ckpt'))
        
    
    def forward(self, observation):
        try:
            port = torch.tensor(list(map(lambda x: x['portfolio'], observation)), dtype=torch.float32, device=self.device)
            cov = torch.tensor(list(map(lambda x: x['covariance'], observation)), dtype=torch.float32, device=self.device)
            action = torch.tensor(list(map(lambda x: x['action'], observation)), dtype=torch.float32, device=self.device)
        except TypeError:
            port = torch.tensor(observation['portfolio'], dtype=torch.float32, device=self.device)
            cov = torch.tensor(observation['covariance'], dtype=torch.float32, device=self.device)
            action = torch.tensor(observation['action'], dtype=torch.float32, device=self.device)
        
        port = port.view((-1, 3, 8, 60))
        cov = cov.view((-1, 3, 8, 8))
        action = action.view((-1, 1, 8, 1)) # shape(1, 1, 8, 1)
        
        port = self.ae.encoder(port)
        port = torch.relu(self.conv_port(port))
        cov = torch.relu(self.conv_cov(cov))
        m = torch.concat([port, cov], dim=1)
        m = torch.relu(self.conv_mix(m)) # shape(1, 20, 8, 1)
        all = torch.concat([m, action], dim=1) # shape(1, 21, 8, 1)
        all = torch.softmax(self.conv_out(all), dim=1)
        
        return all.squeeze()
        

class TD3_Critic(nn.Module):
    def __init__(self, device):
        super(TD3_Critic, self).__init__()
        self.device = device
        self.ae = Autoencoder().to(self.device)
        self.ae.load_state_dict(torch.load('agent/latent_space/Autoencoder.ckpt'))
        
        # Q1
        self.conv_port1 = nn.Conv2d(3, 2, (1, 23))
        self.conv_cov1 = nn.Conv2d(3, 1, (1, 1))
        self.conv_mix1 = nn.Conv2d(3, 20, (1, 8))
        self.conv_out1 = nn.Conv2d(22, 1, 1)
        self.linear1 = nn.Linear(8, 1)
        
        # Q2
        self.conv_port2 = nn.Conv2d(3, 2, (1, 23))
        self.conv_cov2 = nn.Conv2d(3, 1, (1, 1))
        self.conv_mix2 = nn.Conv2d(3, 20, (1, 8))
        self.conv_out2 = nn.Conv2d(22, 1, 1)
        self.linear2 = nn.Linear(8, 1)
    
    def forward(self, observation, act):
        port = torch.tensor(list(map(lambda x: x['portfolio'], observation)), dtype=torch.float32, device=self.device)
        cov = torch.tensor(list(map(lambda x: x['covariance'], observation)), dtype=torch.float32, device=self.device)
        action = torch.tensor(list(map(lambda x: x['action'], observation)), dtype=torch.float32, device=self.device)
        
        port = port.view((-1, 3, 8, 60))
        cov = cov.view((-1, 3, 8, 8))
        action = action.view((-1, 1, 8, 1)) # shape(1, 1, 8, 1)
        act = act.view((-1, 1, 8, 1))
        
        # Q1
        q1_port = self.ae.encoder(port)
        q1_port = torch.relu(self.conv_port1(q1_port))
        q1_cov = torch.relu(self.conv_cov1(cov))
        q1_m = torch.concat([q1_port, q1_cov], dim=1)
        q1_m = torch.relu(self.conv_mix1(q1_m)) # shape(1, 20, 8, 1)
        q1_all = torch.concat([q1_m, action, act], dim=1) # shape(1, 22, 8, 1)
        q1_all = self.conv_out1(q1_all).view((-1, 8))
        q1 = self.linear1(q1_all)
        
        # Q2
        q2_port = self.ae.encoder(port)
        q2_port = torch.relu(self.conv_port2(q2_port))
        q2_cov = torch.relu(self.conv_cov2(cov))
        q2_m = torch.concat([q2_port, q2_cov], dim=1)
        q2_m = torch.relu(self.conv_mix2(q2_m)) # shape(1, 20, 8, 1)
        q2_all = torch.concat([q2_m, action, act], dim=1) # shape(1, 22, 8, 1)
        q2_all = self.conv_out2(q2_all).view((-1, 8))
        q2 = self.linear1(q2_all)
        
        return q1, q2