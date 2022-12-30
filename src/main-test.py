import gym
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
   
model = keras.models.load_model('bossfight-trained-model')
env = gym.make("procgen:procgen-bossfight-v0")
test_episodes = 5

def visualize_filter(model, state):
    successive_outputs = [layer.output for layer in model.layers[1:]]
    visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
    encoded_state = state
    encoded_reshaped = encoded_state.reshape([1, encoded_state.shape[0],  encoded_state.shape[1],  encoded_state.shape[2]])
    successive_feature_maps = visualization_model.predict(encoded_reshaped)
    # Retrieve are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model.layers]
    plt.show()
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        print(feature_map.shape)
        if len(feature_map.shape) == 4:
            
            # Plot Feature maps for the conv / maxpool layers, not the fully-connected layers
        
            n_features = feature_map.shape[-1]  # number of features in the feature map
            size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
            
            # We will tile our images in this matrix
            display_grid = np.zeros((size, size * n_features))
            
            # Postprocess the feature to be visually palatable
            for i in range(n_features):
                x  = feature_map[0, :, :, i]
                x -= x.mean()
                x /= x.std ()
                x *=  64
                x += 128
                x  = np.clip(x, 0, 255).astype('uint8')
                # Tile each filter into a horizontal grid
                display_grid[:, i * size : (i + 1) * size] = x
            # Display the grid
            scale = 20. / n_features
            plt.figure( figsize=(scale * n_features, scale) )
            plt.title ( layer_name )
            plt.grid  ( False )
            plt.imshow( display_grid, aspect='auto', cmap='viridis' )


for episode in range(test_episodes):
    total_training_rewards = 0
    observation = env.reset()
    done = False
    score = 0
    while not done:
        encoded = observation
        encoded_reshaped = encoded.reshape([1, encoded.shape[0],  encoded.shape[1],  encoded.shape[2]])
        predicted = model.predict(encoded_reshaped, verbose=0).flatten()
        # Inspect the filter
        visualize_filter(model, observation)
        action = np.argmax(predicted)
        observation, reward, done, info = env.step(action)

        print(f"reward={reward}")

    print(f"Done test episode {episode}. Score is {score}")