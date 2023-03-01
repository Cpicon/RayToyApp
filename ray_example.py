import gym, ray
from ray.rllib.algorithms import ppo
import numpy as np

class MyEnv(gym.Env):
    def __init__(self, env_config):
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.action_space = gym.spaces.Discrete(2)
    def reset(self):
        return np.array([1])
    def step(self, action):
        return np.array([1]), 1, True, {}


ray.init()
algo = ppo.PPO(env=MyEnv, config={
    "env_config": {},  # config to pass to env class
})

while True:
    print(algo.train())