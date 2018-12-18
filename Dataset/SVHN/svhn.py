from scipy import io
import numpy as np
import random
import tensorflow as tf


class_num = 10
image_size = 32
img_channels = 3

def OneHot(label,n_classes):
    label=np.array(label).reshape(-1)
    label=np.eye(n_classes)[label]

    return label


def prepare_data():
    classes = 10

    data1 = io.loadmat('./data/train_32x32.mat')
    data2 = io.loadmat('./data/test_32x32.mat')
    data3 = io.loadmat('./data/extra_32x32.mat')

    train_data = data1['X']
    train_labels = data1['y']
    test_data = data2['X']
    test_labels = data2['y']
    extra_data = data3['X']
    extra_labels = data3['y']

    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    extra_data = extra_data.astype('float32')

    train_data = np.transpose(train_data, (3, 0, 1, 2))
    test_data = np.transpose(test_data, (3, 0, 1, 2))
    extra_data = np.transpose(extra_data, (3, 0, 1, 2))

    train_labels[train_labels == 10] = 0
    test_labels[test_labels == 10] = 0
    extra_labels[extra_labels == 10] = 0

    train_labels = train_labels[:, 0]
    test_labels = test_labels[:, 0]
    extra_labels = extra_labels[:, 0]

    train_labels = OneHot(train_labels, classes)
    test_labels = OneHot(test_labels, classes)
    extra_labels = OneHot(extra_labels, classes)

    # truncate the train data and test data
    train_data = train_data[0:50000,:,:,:]
    train_labels = train_labels[0:50000,:]
    test_data = test_data[0:10000,:,:,:]
    test_labels = test_labels[0:10000,:]



#    train_data = np.concatenate((train_data,extra_data),axis=0)
#    train_labels = np.concatenate((train_labels,extra_labels),axis=0)

    print('Train data:', train_data.shape, ', Train labels:', train_labels.shape)
    print('Test data:', test_data.shape, ', Test labels:', test_labels.shape)

    return train_data, train_labels, test_data, test_labels


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


def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [32, 32], 4)
    return batch