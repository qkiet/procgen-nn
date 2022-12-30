import gym
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

env = gym.make("procgen:procgen-bossfight-v0", render_mode="rgb_array")
state = env.reset()
print(f"state type is '{type(state)}' shape if possible is '{state.shape}'")
plt.ion()
plt.show()
total_steps = 0
while 1:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    img = state
    imgplot = plt.imshow(img)
    plt.pause(0.05)

