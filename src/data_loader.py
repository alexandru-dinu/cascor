import os
import gzip
import numpy as np


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784, 1)

    return np.array(images, dtype=np.float64), np.array(labels, dtype=np.uint8)


def pre_process(train_imgs, test_imgs):
    avg = np.mean(train_imgs)
    dev = np.std(train_imgs)

    train_imgs -= avg
    train_imgs /= dev
    test_imgs -= avg
    test_imgs /= dev
