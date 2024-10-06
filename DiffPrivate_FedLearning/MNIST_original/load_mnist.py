import tensorflow as tf
import numpy as np
import gzip
import pickle

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Function to save data in gzip format
def save_gzip(filename, data):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(data, f)

# Save training and test data
save_gzip('mnist_train_images.gz', x_train)
save_gzip('mnist_train_labels.gz', y_train)
save_gzip('mnist_test_images.gz', x_test)
save_gzip('mnist_test_labels.gz', y_test)

print("MNIST dataset loaded and saved in gzip format.")
