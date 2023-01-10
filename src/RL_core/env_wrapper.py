import numpy as np

class EnvWrapper:
    def __init__(self, env):
        self.env = env
        self.state = []
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.total_steps = 0

    def reset(self):
        self.total_steps += 1
        return self.env.reset()
    
    def sample_action(self):
        return self.env.action_space.sample()

    def step(self, action):
        self.total_steps += 1
        return self.env.step(action)

    def close(self):
        self.env.close()
