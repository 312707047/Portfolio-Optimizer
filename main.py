from DDPG_agent import DDPG
import gym
import torch

env = gym.make('Pendulum-v1')
env.reset()
# env.render()

params = {
    'env': env,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'GAMMA':0.99,
    'CRITIC_LR':0.001,
    'ACTOR_LR':0.001,
    'TAU': 0.02,
    'BATCH_SIZE':32,
    'MEMORY_SIZE': 100000,
    'EPISODES': 100
}

agent = DDPG(**params)
agent.train()