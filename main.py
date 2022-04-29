from environment.env import TradingEnv
from agent.TD3_agent import TD3

import torch

env = TradingEnv('data', observation_features='All', steps=750, start_date_index=0)


params = {
    'env': env,
    'device': torch.device('cuda'),
    'GAMMA':0.99,
    'CRITIC_LR':0.001,
    'ACTOR_LR':0.00015,
    'TAU_ACTOR': 0.025, # 0.05
    'TAU_CRITIC': 0.001,
    'BATCH_SIZE':64,
    'MEMORY_SIZE': 100000,
    'EPISODES': 1000,
    'POLICY_NOISE': 0.01,
    'NOISE_CLIP': 0.01,
    'POLICY_DELAY': 3,
    'EXPLORATION_NOISE':0.005,
    'EXPLORATION_NOISE_END':0.0005
}


########### training stage 1 ###########
# 
# 
agent = TD3(**params)
# agent.pretrain(pretrain_step=50000)
agent.train(noise='Gaussian')