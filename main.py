#!/usr/bin/env python3

import ast
import sys
import math
import json
import random
import argparse
import numpy as np

try:
    from mnist import MNIST #Reading of datasets
except ImportError:
    print("MNIST could not be imported. Depending on the format of the dataset, this may not be a problem")

class Network:
    def __init__(self, ds_path, batch_size=100, mnist_format=True):
        #Initialise weight matrices
        self.w_i_h1  = np.random.randn(784, 10)
        self.w_h1_h2 = np.random.randn(10, 10)
        self.w_h2_o  = np.random.randn(10, 10)

def sigmoid(x):
    try:
        return 1/(1+(math.e ** -(x)))
    except OverflowError:
        if x > 0:
            return 1
        else:
            return 0

def load_dataset(ds_path, training=True):
    if Network.mnist_format:
        dataset = MNIST(ds_path)
        if training:
            return (dataset.load_training()[0], dataset.train_labels)
        else:
            return (dataset.load_testing()[0], dataset.test_labels)
    else:
        if training:
            with open(ds_path + "/training.json") as training_json:
                dataset = json.load(training_json)
        else:
            with open(ds_path + "/training.json") as testing_json:
                dataset = json.load(testing_json)

        imgs = [ast.literal_eval(img) for img in list(dataset.keys())]
        labels = list(dataset.values())
        return (imgs, labels)

def save_weights(path, weights):
    with open(path, 'w') as f:
        for line in weights:
            np.savetxt(f, line, fmt='%.2f')

def main():
    mat = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    save_weights('weights.txt', mat)

if __name__ == "__main__":
    main()
