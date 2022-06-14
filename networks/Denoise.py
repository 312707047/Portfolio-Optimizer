import numpy as np
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, (1, 3), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, (1, 7), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, (1, 10), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 3, (1, 14), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(3),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 16, (1, 14), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 32, (1, 10), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 64, (1, 7), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, (1, 3), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(3),
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x