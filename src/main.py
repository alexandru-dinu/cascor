import matplotlib.pyplot as plt
import pylab
import numpy as np
import argparse
import pickle

from data_loader import *
from activations import *
from loss_functions import *

from cascor import CascadeCorrelationNet, HiddenUnit

DATA_PATH = "../fashion-mnist/data/fashion"


def _eval(nn, imgs, labels, maximum=0):
    # Compute the confusion matrix
    confusion_matrix = np.zeros((10, 10))
    correct_no = 0
    how_many = imgs.shape[0] if maximum == 0 else maximum
    for i in range(imgs.shape[0])[:how_many]:
        y = np.argmax(nn.forward(imgs[i]))
        t = labels[i]
        if y == t:
            correct_no += 1
        confusion_matrix[y][t] += 1

    return float(correct_no) / float(how_many), confusion_matrix / float(how_many)


def evaluate_net(net, data, e_idx, b_idx, store=None):
    test_acc, test_cm = \
        _eval(net, data["test_imgs"], data["test_labels"])
    train_acc, train_cm = \
        _eval(net, data["train_imgs"], data["train_labels"], 5000)

    print("\t[{}/{}] Train acc: {} ; Test acc: {}".format(e_idx + 1, b_idx + 1, train_acc, test_acc))

    pylab.imshow(test_cm)
    pylab.draw()
    plt.pause(0.001)

    if store is not None:
        store['train_acc'].append(train_acc)
        store['test_acc'].append(test_acc)


def train_batch(net, inputs, labels, args):
    batch_size = inputs.shape[0]
    total_err = 0

    for i in range(batch_size):
        target = np.zeros((10, 1))
        target[labels[i]] = 1

        output = net.forward(inputs[i])

        loss = loss_func(output, target)
        total_err += np.sum(loss)

        net.backward(output - target)
        net.update_parameters(args.learning_rate)

    return total_err


def train_until_no_improvement(net, data, args):
    pylab.ion()

    logs = {
        'train_acc': [],
        'test_acc': []
    }

    cnt = 0

    total_error = 0
    num_batches = data['num_train'] // args.batch_size

    # TODO: test
    # num_epochs = args.num_epochs
    num_epochs = net.num_hidden_units()
    # TODO: test

    for e_idx in range(num_epochs + 1):
        # shuffle training data
        shuffled_idx = np.random.permutation(data['num_train'])

        # # TODO tweak
        # # should decrease after each epoch
        # eps = 1.5 - (e_idx / (data['num_epochs'] - 1)) * 1.0
        # # should increase after each epoch
        # window_size = 2 + e_idx
        # errors = [0.0] * window_size

        # print("epoch [{}]; eps/window_size = {}/{}".format(e_idx, eps, window_size))

        # for each batch
        for b_idx in range(num_batches):
            current_slice = shuffled_idx[b_idx * args.batch_size: (b_idx + 1) * args.batch_size]
            batch_inputs = data['train_imgs'][current_slice]
            batch_labels = data['train_labels'][current_slice]

            batch_error = train_batch(net, batch_inputs, batch_labels, args)
            total_error += batch_error

            # errors[cnt % window_size] = batch_error

            # evaluate the network each batch
            evaluate_net(net, data, e_idx, b_idx, logs)
            # print(max(errors) - min(errors))

            # no significant improvement
            # if max(errors) - min(errors) < eps:
            #     print("Error doesn't improve -> insert a new hidden unit")
            #     return total_error, logs

            cnt += 1

    # all epochs have finished
    return total_error, logs


def test(net, data, args):
    if args.test_shuffled:
        test_idxs = np.random.permutation(len(data['test_imgs']))
        train_idxs = np.random.permutation(len(data['train_imgs']))
    else:
        test_idxs = np.arange(len(data['test_imgs']))
        train_idxs = np.arange(len(data['train_imgs']))

    test_batch_size = 1000
    train_batch_size = 5000

    test_correct = 0
    train_correct = 0

    test_slice = test_idxs[:test_batch_size]
    test_batch = data['test_imgs'][test_slice]
    test_labels = data['test_labels'][test_slice]

    for j in range(test_batch_size):
        out = net.forward(test_batch[j])
        if np.argmax(out) == test_labels[j]:
            test_correct += 1

    train_slice = train_idxs[:train_batch_size]
    train_batch = data['train_imgs'][train_slice]
    train_labels = data['train_labels'][train_slice]

    for j in range(train_batch_size):
        out = net.forward(train_batch[j])
        if np.argmax(out) == train_labels[j]:
            train_correct += 1

    train_acc = (1.0 * train_correct / train_batch_size)
    test_acc = (1.0 * test_correct / test_batch_size)

    return train_acc, test_acc


def train(net, data, args):
    HISTORY = []
    RESULTS = {'train': [], 'test': []}

    hu_idx = 0
    eps = 1

    # train the network until the error doesn't change anymore
    print(net.to_string())
    error, logs = train_until_no_improvement(net, data, args)
    HISTORY.append(logs)

    train_acc, test_acc = test(net, data, args)
    RESULTS['train'].append(train_acc)
    RESULTS['test'].append(test_acc)
    # ---

    prev_error = 0

    while net.num_hidden_units() < args.max_hidden_units:
        # construct new hidden unit
        next_num_inputs = net.num_hidden_units() + net.num_inputs
        h = HiddenUnit(hu_idx, next_num_inputs, net.num_outputs, logistic)
        hu_idx += 1

        # maximize correlation for the current hidden unit
        net.maximize_correlation(data['train_imgs'], data['train_labels'], h)

        # insert hidden unit in the network
        # this means that hu's input weights are frozen
        net.insert_hidden_unit(h)

        prev_error = error

        # resume training, but now consider the previously added hidden unit
        print(net.to_string())
        error, logs = train_until_no_improvement(net, data, args)
        HISTORY.append(logs)

        train_acc, test_acc = test(net, data, args)
        RESULTS['train'].append(train_acc)
        RESULTS['test'].append(test_acc)
        # ---

    # dump data
    for i, r in enumerate(HISTORY):
        pickle.dump(r, open("acc_" + str(i), "wb"))

    pickle.dump(RESULTS, open("results_" + str(args.max_hidden_units), "wb"))


def main(args):
    train_imgs, train_labels = load_mnist(DATA_PATH, kind='train')
    test_imgs, test_labels = load_mnist(DATA_PATH, kind='t10k')

    pre_process(train_imgs, test_imgs)

    data = {
        'train_imgs': train_imgs,
        'train_labels': train_labels,
        'test_imgs': test_imgs,
        'test_labels': test_labels,
        'num_train': len(train_imgs),
    }

    # construct network
    net = CascadeCorrelationNet(784, 10, logistic)

    train(net, data, args)


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()

    argparse.add_argument(
        '--learning_rate', dest='learning_rate', default=0.1, type=float
    )
    argparse.add_argument(
        '--eval_every', dest='eval_every', default=1000, type=int
    )
    argparse.add_argument(
        '--num_epochs', dest='num_epochs', default=1, type=int
    )
    argparse.add_argument(
        '--batch_size', dest='batch_size', default=1000, type=int
    )
    argparse.add_argument(
        '--max_hidden_units', dest='max_hidden_units', default=5, type=int
    )
    argparse.add_argument(
        "--test_shuffled", dest="test_shuffled", action="store_true",
    )

    args = argparse.parse_args()

    main(args)
