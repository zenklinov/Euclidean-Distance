pip install tensorflow-gpu==2.12.0 #Skip if already installed

import tensorflow as tf
tf.config.list_physical_devices('GPU')

pip install fastdtw #Skip if already installed

from fastdtw import fastdtw 

import os
import pickle
import numpy as np
import random

# Assume dataku is defined before this code

# Selecting only temporal columns
temporal_data = data.iloc[:, 2:]

# Converting temporal data to numpy array
data_array = temporal_data.to_numpy()

# Calculating the number of rows and columns in data_array
n_rows, n_cols = data_array.shape

# Renaming variable to dtw_distances
dtw_distances = np.zeros((n_rows, n_rows))

# Function to save progress
def save_progress(i, j_iter, dtw_distances):
    save_directory = "/content/drive/My Drive/"

    # Create directory if not exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    save_path = os.path.join(save_directory, "dtw_distances.pkl")

    # Save progress along with i, j, and dtw_distances
    progress = (i * n_rows + j_iter) / (n_rows * n_rows) * 100
    with open(save_path, "wb") as file:
        pickle.dump((progress, i, j_iter, dtw_distances), file)
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
    load_path = "/content/drive/My Drive/dtw_distances.pkl"
    return load_progress_safe(load_path, n_rows)

# Load the last progress or start from the beginning if no previous progress exists
progress, i, j_iter, dtw_distances = load_progress()

# Initialize j_iter outside the outer loop
j_iter = 0

# TensorFlow session
tf.reset_default_graph()
sess = tf.InteractiveSession()

# Placeholder for data
data_placeholder = tf.placeholder(tf.float32, shape=(None, n_cols))

# Calculate DTW distance using TensorFlow
def dtw_distance(x, y):
    n, m = tf.shape(x)[0], tf.shape(y)[0]
    dtw = tf.Variable(tf.zeros(shape=(n+1, m+1)))
    for i in range(1, n+1):
        for j in range(1, m+1):
            dtw_cost = tf.abs(tf.reduce_sum(tf.subtract(tf.expand_dims(x[i-1], 0), tf.expand_dims(y[j-1], 0))))
            dtw_cost += tf.minimum(tf.minimum(dtw[i-1, j], dtw[i, j-1]), dtw[i-1, j-1])
            dtw[i, j].assign(dtw_cost)
    return dtw[n, m]

# Loop to calculate DTW distance between rows
try:
    for i_iter in range(i, n_rows):
        # Set initial value for j_iter based on i_iter value
        j_start = 0 if i_iter == i else 0

        # Loop to calculate DTW distance with row j
        for j_iter in range(j_start, n_rows):
            # If j_iter is equal to i_iter, we don't need to calculate DTW with itself
            if j_iter != i_iter:
                # Calculate DTW distance using TensorFlow
                distance = dtw_distance(data_array[i_iter:i_iter+1, :], data_array[j_iter:j_iter+1, :])
                
                # Fill the DTW distance matrix
                dtw_distances[i_iter, j_iter] = distance
                dtw_distances[j_iter, i_iter] = distance  # Making it symmetric

                # Save progress every time finishing calculating one pair of rows
                save_progress(i_iter, j_iter, dtw_distances)

        # Print progress only for some random row pairs
        if random.random() < 0.01:  # Change 0.01 according to your needs
            progress = (i_iter * n_rows + j_iter) / (n_rows * n_rows) * 100
            print(f"Process reached row {i_iter+1} ({progress:.2f}% completed)")

except KeyboardInterrupt:
    print("\nInterrupt detected. Saving progress...")
    # Save progress when interrupted
    save_progress(i_iter, j_iter, dtw_distances)
    print("Progress successfully saved. Please continue execution to resume calculation.")

# Close TensorFlow session
sess.close()
