import numpy as np
import torch
import torch.nn as nn

class Encoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        '''input shape: (3*8*60)'''
        self.conv_1 = nn.Conv2d(3, 64, (1, 3), padding=0)
        self.conv_2 = nn.Conv2d(64, 32, (1, 7), padding=0)
        self.conv_3 = nn.Conv2d(32, 16, (1, 10), padding=0)
        self.conv_4 = nn.Conv2d(16, 3, (1, 14), padding=0)
        
        self.bn_1 = nn.BatchNorm2d(64)
        self.bn_2 = nn.BatchNorm2d(32)
        self.bn_3 = nn.BatchNorm2d(16)
        self.bn_4 = nn.BatchNorm2d(3)
    
    def forward(self, x):
        x = torch.relu(self.conv_1(x))
        x = self.bn_1(x)
        x = torch.relu(self.conv_2(x))
        x = self.bn_2(x)
        x = torch.relu(self.conv_3(x))
        x = self.bn_3(x)
        x = torch.relu(self.conv_4(x))
        x = self.bn_4(x)
        
        return x


class Decoder(nn.Module):
    
    def __init__(self):
        super(Decoder, self).__init__()
        '''input shape: (3*8*30)'''
        self.convt_1 = nn.ConvTranspose2d(3, 16, (1, 14), padding=0)
        self.convt_2 = nn.ConvTranspose2d(16, 32, (1, 10), padding=0)
        self.convt_3 = nn.ConvTranspose2d(32, 64, (1, 7), padding=0)
        self.convt_4 = nn.ConvTranspose2d(64, 3, (1, 3), padding=0)
        
        self.bn_1 = nn.BatchNorm2d(3)
        self.bn_2 = nn.BatchNorm2d(16)
        self.bn_3 = nn.BatchNorm2d(32)
        self.bn_4 = nn.BatchNorm2d(64)
    
    def forward(self, x):
        x = torch.relu(self.convt_1(x))
        x = self.bn_2(x)
        x = torch.relu(self.convt_2(x))
        x = self.bn_3(x)
        x = torch.relu(self.convt_3(x))
        x = self.bn_4(x)
        x = torch.relu(self.convt_4(x))
        x = self.bn_1(x)
        
        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x


class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
            (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)