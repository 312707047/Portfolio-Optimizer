import torch
import torch.nn as nn

class Encoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        '''input shape: (3*7*60)'''
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, (1, 3), padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, (1, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, (1, 3), padding=1),
            nn.ReLU()
        )
        
        self.flatten = nn.Flatten(1)