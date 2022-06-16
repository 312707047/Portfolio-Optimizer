from environment.env import TradingEnv
from environment.wrappers import env_wrapper
from agent.TD3_agent import TD3


env = TradingEnv('data', commission=0.000000001, time_cost=0.05, steps=800, start_date_index=0, augment=0.05)
# train: 800, test: 244
env = env_wrapper(env)

params = {
    'env': env,
    'device': 'cuda',
    'GAMMA':0.96,
    'CRITIC_LR':0.0001,
    'ACTOR_LR':0.00005,
    'TAU_ACTOR': 0.05, # 0.05
    'TAU_CRITIC': 0.05,
    'BATCH_SIZE':64,
    'MEMORY_SIZE': 100000,
    'EPISODES': 1000,
    'POLICY_NOISE': 0.025,
    'NOISE_CLIP': 0.05,
    'POLICY_DELAY': 3,
    'EXPLORATION_NOISE':0.0075,
    'EXPLORATION_NOISE_END':0.0075,
    'print_info': True
}


########### training stage 1 ###########
# 
# 
agent = TD3(**params)
# agent.pretrain(pretrain_step=50000)
# agent.load_model('networks/saved_models/another_stage')
# agent.train()
agent.test(model_path='networks/saved_models/another_stage')