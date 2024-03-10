pip install tensorflow-gpu==2.12.0 #Skip if already installed

import tensorflow as tf
tf.config.list_physical_devices('GPU')

import os
import pickle
import numpy as np
import random

# Assume dataku is defined before this code

# Selecting only temporal columns
temporal_data = dataku.iloc[:, 2:]

# Converting temporal data to numpy array
data_array = temporal_data.to_numpy()

# Calculating the number of rows and columns in data_array
n_rows, n_cols = data_array.shape

# Renaming variable to euclidean_distances
euclidean_distances = np.zeros((n_rows, n_rows))

# Function to save progress
def save_progress(i, j_iter, distances, save_directory="/content/drive/My Drive/"):
    # Create directory if not exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    save_path = os.path.join(save_directory, "euclidean_distance.pkl")

    # Save progress along with i, j, and distances
    progress = (i * n_rows + j_iter) / (n_rows * n_rows) * 100
    with open(save_path, "wb") as file:
        pickle.dump((progress, i, j_iter, distances), file)
    print(f"Progress saved at iteration {i}, j_iter {j_iter}")

# Function to load progress with EOFError handling
def load_progress_safe(load_path, n_rows):
    try:
        with open(load_path, "rb") as file:
            # Check if the file is empty
            if os.path.getsize(load_path) > 0:
                data = pickle.load(file)
                return data
            else:
                # If the file is empty, return default values
                return 0, 0, 0, np.zeros((n_rows, n_rows))
    except (FileNotFoundError, EOFError):
        # Handle both FileNotFoundError and EOFError
        return 0, 0, 0, np.zeros((n_rows, n_rows))

# Define the load_progress function
def load_progress():
    load_path = "/content/drive/My Drive/euclidean_distance.pkl"
    return load_progress_safe(load_path, n_rows)

# Load the last progress or start from the beginning if no previous progress exists
progress, i, j_iter, euclidean_distances = load_progress()

# Initialize j_iter outside the outer loop
j_iter = 0

# TensorFlow session
tf.reset_default_graph()
sess = tf.InteractiveSession()

# Placeholder for data
data_placeholder = tf.placeholder(tf.float32, shape=(None, n_cols))

# Calculate Euclidean distance using TensorFlow
euclidean_distances_tf = tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(data_placeholder, 1) - tf.expand_dims(data_placeholder, 0)), axis=2))

try:
    # Loop to calculate Euclidean distance between rows
    for i_iter in range(i, n_rows):
        # Set initial value for j_iter based on i_iter value
        j_start = 0 if i_iter == i else 0

        # Calculating Euclidean distance
        for j_iter in range(j_start, n_rows):
            distance = sess.run(euclidean_distances_tf, feed_dict={data_placeholder: [data_array[i_iter], data_array[j_iter]]})[0, 1]
            euclidean_distances[i_iter, j_iter] = distance
            euclidean_distances[j_iter, i_iter] = distance  # Utilizing symmetry property

        # Save progress every time finishing calculating one row
        save_progress(i_iter, j_iter, euclidean_distances)

        # Print progress only for some random row pairs
        if random.random() < 0.01:  # Change 0.01 according to your needs
            progress = (i_iter * n_rows + j_iter) / (n_rows * n_rows) * 100
            print(f"Process reached row {i_iter+1} ({progress:.2f}% completed)")

except KeyboardInterrupt:
    print("\nInterrupt detected. Saving progress...")
    # Save progress when interrupted
    save_progress(i_iter, j_iter, euclidean_distances)
    print("Progress successfully saved. Please continue execution to resume calculation.")

# Close TensorFlow session
sess.close()
