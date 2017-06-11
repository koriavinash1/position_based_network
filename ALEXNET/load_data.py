import tensorflow as tf
import os
import numpy as np
import cv2

train_images_path = "/media/koriavinash/New Volume/Research/MNIST_database/newMNIST/train-images/"
test_images_path = "/media/koriavinash/New Volume/Research/MNIST_database/newMNIST/test-images/"
train_labels_path = "/media/koriavinash/New Volume/Research/MNIST_database/newMNIST/labels/trainlabels.csv"
test_labels_path = "/media/koriavinash/New Volume/Research/MNIST_database/newMNIST/labels/testlabels.csv"


def extract_images(folder_path, total_images):
    return [ cv2.imread(folder_path + str(i) + ".jpg") for i in xrange(total_images)]

def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    labels_one_hot = np.zeros((num_labels, num_classes))
    for i in xrange(len(labels_dense)):
        labels_one_hot[i][int(labels_dense[i][1])] = 1
    return labels_one_hot

def extract_labels(file_path, one_hot = False):
    labels = np.genfromtxt(file_path, delimiter=",")
    if one_hot:
        return dense_to_one_hot(labels)
    return labels

# print extract_labels(test_labels_path, one_hot = True)
# print extract_images(test_images_path, 10)


class DataSet(object):
    def __init__(self, images, labels):
        self._num_examples = len(images)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def shuffle_in_unison(self, a, b):
        a = np.array(a)
        b = np.array(b, ndmin=2)
        assert len(a) == len(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
        return shuffled_a, shuffled_b

    def next_batch(self, batch_size):
        if self._index_in_epoch > len(self._images)-batch_size:
            self._index_in_epoch = 0
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        return self.shuffle_in_unison(self._images[start:end], self._labels[start:end])


def read_data_sets(one_hot=False, train_image_number = 360000, test_image_number = 60000):
    class DataSets(object):
        pass
    data_sets = DataSets()
    VALIDATION_SIZE = 5000
    print "extracting training images...."
    train_images = extract_images(train_images_path, train_image_number)
    print "extracting training labels...."
    train_labels = extract_labels(train_labels_path, one_hot=one_hot)
    print "extracting testing images...." 
    test_images = extract_images(test_images_path, test_image_number)
    print "extracting testing labels...."
    test_labels = extract_labels(test_labels_path, one_hot=one_hot)
    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]
    data_sets.train = DataSet(train_images, train_labels)
    data_sets.validation = DataSet(validation_images, validation_labels)
    data_sets.test = DataSet(test_images, test_labels)
    return data_sets