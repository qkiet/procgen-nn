import gym
import tensorflow as tf
import numpy as np
from tensorflow import keras
   
model = keras.models.load_model('trained-model-180')
env = gym.make('CartPole-v1', render_mode="human")
test_episodes = 5

for episode in range(test_episodes):
    total_training_rewards = 0
    observation = env.reset()[0]
    done = False
    score = 0
    while not done:
        score += 1
        encoded = observation
        encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
        predicted = model.predict(encoded_reshaped, verbose=0).flatten()
        action = np.argmax(predicted)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    print(f"Done test episode {episode}. Score is {score}")