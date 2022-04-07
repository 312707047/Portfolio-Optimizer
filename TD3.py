from TD3_agent import TD3
import gym
import torch

env = gym.make('Pendulum-v1')
env.reset()
# env.render()

params = {
    'env': env,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'GAMMA':0.96,
    'CRITIC_LR':0.001,
    'ACTOR_LR':0.001,
    'TAU': 0.05,
    'BATCH_SIZE':256,
    'MEMORY_SIZE': 100000,
    'EPISODES': 1000,
    'POLICY_NOISE': 0.2,
    'NOISE_CLIP': 0.5,
    'POLICY_DELAY': 2,
    'EXPLORATION_NOISE':0.1
}

agent = TD3(**params)
agent.train(noise='Gaussian')