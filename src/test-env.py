import gym

env = gym.make("procgen:procgen-bossfight-v0", render_mode="human")
state = env.reset()
total_steps = 0
while 1:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action) 
