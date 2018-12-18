import copy
import os
from subprocess import call

import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.linear_model

import h5py
import pickle
import random

class_num = 100
image_size = 32
img_channels = 3

def download_data():
    print("Downloading...")
    if not os.path.exists("cifar-100-python.tar.gz"):
        call(
            "wget http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
            shell=True
        )
        print("Downloading done.\n")
    else:
        print("Dataset already downloaded. Did not download twice.\n")
    print("Extracting...")
    cifar_python_directory = os.path.abspath("cifar-100-python")
    if not os.path.exists(cifar_python_directory):
        call(
            "tar -zxvf cifar-100-python.tar.gz",
            shell=True
        )
        print("Extracting successfully done to {}.".format(cifar_python_directory))
    else:
        print("Dataset already extracted. Did not extract twice.\n")


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_data_one(file):
    batch = unpickle(file)
    data = batch[b'data']
    labels = batch[b'labels']
    print("Loading %s : %d." % (file, len(data)))
    return data, labels


def load_data(files, data_dir, label_count):
    global image_size, img_channels
    data, labels = load_data_one(data_dir + '/' + files[0])
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + '/' + f)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
    labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    data = data.reshape([-1, img_channels, image_size, image_size])
    data = data.transpose([0, 2, 3, 1])
    return data, labels


def load_data(train_file):
        data = []
        labels = []
        d = unpickle(train_file)
        data = d[b'data']
        coarse_labels = d[b'coarse_labels']
        fine_labels = d[b'fine_labels']
        length = len(d[b'fine_labels'])

        return (
            np.moveaxis(data.reshape(length, 3, 32, 32),1,-1),
            np.array(coarse_labels),
            np.array(fine_labels)
        )


def prepare_data():
    print("======Loading data======")
    download_data()
    data_dir = './cifar-100-python'
    image_dim = image_size * image_size * img_channels

    train_data, train_labels_coarse, train_labels_fine = load_data(data_dir + '/train')
    test_data, test_labels_coarse, test_labels_fine = load_data(data_dir + '/test')

    print("Train data:", np.shape(train_data), np.shape(train_labels_fine))
    print("Test data :", np.shape(test_data), np.shape(test_labels_fine))
    print("======Load finished======")

    print("======Shuffling data======")
    indices = np.random.permutation(len(train_data))
    train_data = train_data[indices]
    train_labels = train_labels_fine[indices]
    test_labels = test_labels_fine
    print("======Prepare Finished======")

    return train_data, train_labels, test_data, test_labels


def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - np.mean(x_test[:, :, :, 0])) / np.std(x_test[:, :, :, 0])
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - np.mean(x_test[:, :, :, 1])) / np.std(x_test[:, :, :, 1])
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - np.mean(x_test[:, :, :, 2])) / np.std(x_test[:, :, :, 2])

    return x_train, x_test


def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [32, 32], 4)
    return batch


def encode(y, class_num):
    y_matrix = []
    for i in range(len(y)):
        y_row = []
        for j in range(class_num):
            if y[i] == j:
                y_row.append(1)
            else:
                y_row.append(0)
        y_matrix.append(y_row)
    return y_matrix