"""
A Minimal Deep Q-Learning Implementation (minDQN)
Running this code will render the agent solving the CartPole environment using OpenAI gym. Our Minimal Deep Q-Network is approximately 150 lines of code. In addition, this implementation uses Tensorflow and Keras and should generally run in less than 15 minutes.
Usage: python3 minDQN.py
"""

import gym
import numpy as np
from RL_core.model import create_agent, train, encode_observation
import RL_core.env_wrapper as en_wrapper

from collections import deque
import time
import sys

"""
    Usage: python <this script> <number of training episodes>
"""

# An episode a full game
train_episodes = int(sys.argv[1])
    
def main():
    RANDOM_SEED = 5

    env_origin = gym.make(
        "procgen:procgen-bossfight-v0", 
        distribution_mode="easy"
        )
    wrapping_env = en_wrapper.EnvWrapper(env_origin)
    
    np.random.seed(RANDOM_SEED)

    print("Action Space: {}".format(wrapping_env.action_space))
    print("State space: {}".format(wrapping_env.observation_space))
    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1 # You can't explore more than 100% of the time
    min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
    decay = 0.01
    # 1. Initialize the Target and Main models
    # Main Model (updated every 4 steps)
    model = create_agent(wrapping_env)
    # Target Model (updated every 100 steps)
    target_model = create_agent(wrapping_env)
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=50_000)

    target_update_counter = 0

    # X = states, y = actions
    X = []
    y = []

    steps_to_update_target_model = 0

    for episode in range(train_episodes):
        total_training_rewards = 0
        observation = wrapping_env.reset()
        done = False
        total_step_of_episode = 0
        ts_episode_begin = time.time()
        while not done:
            steps_to_update_target_model += 1
            total_step_of_episode += 1

            random_number = np.random.rand()
            # 2. Explore using the Epsilon Greedy Exploration Strategy
            if random_number <= epsilon:
                # Explore
                action = wrapping_env.sample_action()
            else:
                # Exploit best known action
                # model dims are (batch, env.observation_space.n)
                predicted = model.predict(encode_observation(observation), verbose=0).flatten()
                action = np.argmax(predicted)
            new_observation, reward, done, info = wrapping_env.step(action)
            replay_memory.append([observation, action, reward, new_observation, done])

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 4 == 0 or done:
                train(wrapping_env, replay_memory, model, target_model, done)

            observation = new_observation
            total_training_rewards += reward

            if done:
                if steps_to_update_target_model >= 100:
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break
        ts_episode_end = time.time()
        print(f"End train episode {episode} with {total_step_of_episode} episodes and {ts_episode_end-ts_episode_begin}s")
        print(f"Total steps so far: {wrapping_env.total_steps}")
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
    wrapping_env.close()
    model.save('bossfight-trained-model')

if __name__ == '__main__':
    main()