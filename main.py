from environment.env import TradingEnv
from environment.wrappers import env_wrapper
from agent.TD3_agent import TD3


env = TradingEnv('data', commission=0.01, steps=750, start_date_index=0)
env = env_wrapper(env)

params = {
    'env': env,
    'device': 'cuda',
    'GAMMA':0.96,
    'CRITIC_LR':0.001,
    'ACTOR_LR':0.0005,
    'TAU_ACTOR': 0.05, # 0.05
    'TAU_CRITIC': 0.05,
    'BATCH_SIZE':64,
    'MEMORY_SIZE': 100000,
    'EPISODES': 100,
    'POLICY_NOISE': 0.0025,
    'NOISE_CLIP': 0.005,
    'POLICY_DELAY': 3,
    'EXPLORATION_NOISE':0.05,
    'EXPLORATION_NOISE_END':0.005,
    'print_info': False
}


########### training stage 1 ###########
# 
# 
agent = TD3(**params)
# agent.pretrain(pretrain_step=50000)
agent.train()