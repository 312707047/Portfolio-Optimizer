from environment.env import TradingEnv
from agent.TD3_agent import TD3

import torch

env = TradingEnv('data', observation_features='All')


params = {
    'env': env,
    'device': torch.device('cpu'),
    'GAMMA':0.96,
    'CRITIC_LR':0.001,
    'ACTOR_LR':0.001,
    'TAU': 0.05,
    'BATCH_SIZE':16,
    'MEMORY_SIZE': 100000,
    'EPISODES': 1000,
    'POLICY_NOISE': 0.2,
    'NOISE_CLIP': 0.5,
    'POLICY_DELAY': 2,
    'EXPLORATION_NOISE':0.1
}

agent = TD3(**params)
agent.train(noise='Gaussian')
