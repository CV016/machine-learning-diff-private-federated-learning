#!/bin/sh
STRING="Loading the MNIST dataset and creating clients"
echo $STRING

# Change to the DiffPrivate_FedLearning directory
cd DiffPrivate_FedLearning || { echo "Directory not found"; exit 1; }

# Create a directory for MNIST if it doesn't exist
mkdir -p MNIST_original
cd MNIST_original || exit 1

# Create a Python script to load and save the MNIST dataset in gzip format
cat <<EOF > load_mnist.py
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
EOF

# Run the Python script to load and save the MNIST dataset
python3 load_mnist.py

# Change back to the parent directory to run Create_clients.py
cd ../.. || exit 1  # Go back to the parent directory of DiffPrivate_FedLearning

# Run the Create_clients.py script
python3 Create_clients.py 

STRING2="You can now run differentially private federated learning on the MNIST data set. Type python sample.py --h for help"
echo $STRING2
STRING3="An example: python sample.py --N 100 would run differentially private federated learning on 100 clients for a privacy budget of (epsilon = 8, delta = 0.001)"
echo $STRING3
STRING4="For more information on how to use the functions please refer to their documentation"
echo $STRING4
