import torch
import os
import pandas as pd
import numpy as np

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=20):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t/self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


class LinearAnneal:
    """Decay a parameter linearly"""
    def __init__(self, start_val, end_val, steps):
        self.p = start_val
        self.end_val = end_val
        self.decay_rate = (start_val - end_val) / steps

    def anneal(self):
        if self.p > self.end_val:
            self.p -= self.decay_rate
        return self.p


def output_recorder(model_type, output, layer_name, path='output_records'):
    if output.shape[0] != 1:
        return
    
    if not os.path.isdir(path):
        os.mkdir(path)
    
    if not os.path.isdir(os.path.join(path, model_type)):
        os.mkdir(os.path.join(path, model_type))
    
    if not os.path.isdir(os.path.join(path, model_type, layer_name)):
        os.mkdir(os.path.join(path, model_type, layer_name))
    
    num_data = output.shape[1]
    for i in range(num_data):
        df = pd.DataFrame(np.array(output[0][i].squeeze().detach().cpu())).T
        
        if os.path.isfile(os.path.join(path, model_type, layer_name)+'\\'+f'output_{i}.csv') == True:
            df.to_csv(os.path.join(path, model_type, layer_name)+'\\'+f'output_{i}.csv', mode='a', index=False, header=False)
        else:
            df.to_csv(os.path.join(path, model_type, layer_name)+'\\'+f'output_{i}.csv', index=False)


def weight_recorder(model_type, layer, layer_name, path='weight_records'):
    
    # if not os.path.isdir(os.path.join(path, layer_name)):
    #     os.mkdir(os.path.join(path, layer_name))
    
    df = pd.DataFrame(np.array(layer.weight.data.squeeze().detach().cpu())).T
    
    if os.path.isfile(path+'\\'+f'{layer_name}_weight.csv') == True:
        df.to_csv(os.path.join(path, layer_name)+'\\'+f'output_.csv', mode='a', index=False, header=False)
    else:
        df.to_csv(os.path.join(path, layer_name)+'\\'+f'output_.csv', index=False)