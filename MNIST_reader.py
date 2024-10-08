import os
import pickle
import numpy as np
import gzip

def read(dataset="training", path=""):
    if dataset == "training":
        fname_img = os.path.join(path, 'mnist_train_images.gz')
        fname_lbl = os.path.join(path, 'mnist_train_labels.gz')
    elif dataset == "testing":
        fname_img = os.path.join(path, 'mnist_test_images.gz')
        fname_lbl = os.path.join(path, 'mnist_test_labels.gz')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    print(f"Reading labels from: {fname_lbl}")
    print(f"Reading images from: {fname_img}")

    # Load label data from gzip file
    with gzip.open(fname_lbl, 'rb') as flbl:
        lbl = pickle.load(flbl)

    # Load image data from gzip file
    with gzip.open(fname_img, 'rb') as fimg:
        img = pickle.load(fimg)

    # Reshape and normalize the image data if necessary
    img = img.astype(float) / 255.0  # Normalize image data to [0, 1]

    return img, lbl


def get_data(d):
    # Load the data
    x_train, y_train = read('training', 'DiffPrivate_FedLearning/MNIST_original/')
    x_test, y_test = read('testing', 'DiffPrivate_FedLearning/MNIST_original/')

    # Create validation set
    x_vali = list(x_train[50000:].astype(float))
    y_vali = list(y_train[50000:].astype(float))

    # Reduce training set to 50000 samples
    x_train = x_train[:50000].astype(float)
    y_train = y_train[:50000].astype(float)

    # Sort training set by labels to make federated learning non i.i.d.
    indices_train = np.argsort(y_train)
    sorted_x_train = list(x_train[indices_train])
    sorted_y_train = list(y_train[indices_train])

    # Prepare test set
    x_test = list(x_test.astype(float))
    y_test = list(y_test.astype(float))

    return sorted_x_train, sorted_y_train, x_vali, y_vali, x_test, y_test


class Data:
    def __init__(self, save_dir, n):
        raw_directory = save_dir + '/DATA'
        self.client_set = pickle.load(open(raw_directory + '/clients/' + str(n) + '_clients.pkl', 'rb'))
        self.sorted_x_train, self.sorted_y_train, self.x_vali, self.y_vali, self.x_test, self.y_test = get_data(save_dir)
