import torch
import torch.nn as nn
import torch.nn.functional as F


class TD3_Actor(nn.Module):
    def __init__(self, device):
        super(TD3_Actor, self).__init__()
        self.device = device

        self.conv_port = nn.Conv2d(3, 2, (1, 3))
        self.conv_mix = nn.Conv2d(2, 20, (1, 58))
        self.conv_out = nn.Conv2d(21, 1, 1)
    
    def forward(self, observation):
        
        port = torch.tensor(observation['observation'], dtype=torch.float32, device=self.device)
        action = torch.tensor(observation['action'], dtype=torch.float32, device=self.device)
        
        port = port.view(-1, 3, 8, 60)
        action = action.view((-1, 1, 8, 1))
            
        port = F.leaky_relu(self.conv_port(port))
        m = F.leaky_relu(self.conv_mix(port)) # shape(1, 20, 8, 1)
        all = torch.concat([m, action], dim=1) # shape(1, 21, 8, 1)
        all = self.conv_out(all)
        
        return all.squeeze()
        

class TD3_Critic(nn.Module):
    def __init__(self, device):
        super(TD3_Critic, self).__init__()
        self.device = device
    
        # Q1
        self.conv_port1 = nn.Conv2d(3, 2, (1, 3))
        self.conv_mix1 = nn.Conv2d(2, 20, (1, 58))
        self.conv_out1 = nn.Conv2d(22, 1, (8, 1))
        
        # Q2
        self.conv_port2 = nn.Conv2d(3, 2, (1, 3))
        self.conv_mix2 = nn.Conv2d(2, 20, (1, 58))
        self.conv_out2 = nn.Conv2d(22, 1, (8, 1))
    
    def forward(self, observation, act):

        port = torch.tensor(observation['observation'], dtype=torch.float32, device=self.device)
        action = torch.tensor(observation['action'], dtype=torch.float32, device=self.device)
        
        port = port.view(-1, 3, 8, 60)
        action = action.view((-1, 1, 8, 1))
        act = act.view((-1, 1, 8, 1))
    
        # Q1
        q1_port = F.leaky_relu(self.conv_port1(port))
        q1_port = F.leaky_relu(self.conv_mix1(q1_port)) # shape(1, 20, 8, 1)
        q1_port = torch.concat([q1_port, action, act], dim=1) # shape(1, 22, 8, 1)
        q1_port = self.conv_out1(q1_port).squeeze()
        
        # Q2
        q2_port = F.leaky_relu(self.conv_port2(port))
        q2_port = F.leaky_relu(self.conv_mix2(q2_port)) # shape(1, 20, 8, 1)
        q2_port = torch.concat([q2_port, action, act], dim=1) # shape(1, 22, 8, 1)
        q2_port = self.conv_out2(q2_port).squeeze()
            
        return q1_port, q2_port