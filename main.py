#!/usr/bin/env python3

import ast
import math
import json
import numpy as np

try:
    from mnist.loader import MNIST #Reading of datasets
except ImportError:
    print("MNIST could not be imported. Depending on the format of the dataset, this may not be a problem")

class Network:

    def __init__(self, ds_path, mnist_format, weights, batch_size):
        #Initialise weights
        if weights:
            self.weights = weights
        else:
            self.weights = [
                np.random.randn(784, 10), #Input -> h1
                np.random.randn(10, 10),  #h1    -> h2
                np.random.randn(10, 10)   #h2    -> Output
            ]

        self.biases = np.random.uniform(-1, 1, 30)

        #Load datasets
        raw_datasets  = load_dataset(ds_path, True, mnist_format)
        sets, labels  = raw_datasets[0], raw_datasets[1]
        self.batches = [] #[{item1: label1, item_2: label_2}, {item_101: label_101}]
        for i in range(0, math.floor(len(sets) / batch_size)):
            batch = {}
            for j in range(batch_size):
                batch[tuple(sets[j])] = labels[j]
            self.batches.append(batch)

    def activate(self, sample):
        activations_prior = [] #Store last layer activations
        for i, data in zip(range(784), sample): #Activate input layer
            activations_prior[i] = data

        for i in range(3):
            activations_curr = []
            for neuron in range(10):
                activation = 0
                for weight in self.weights[i]:
                    activation += (activations_prior[neuron] * weight)
                activations_curr.append(sigmoid(activation))
            activations_prior = list(activations_curr)

        return activations_prior

def sigmoid(x):
    try:
        return 1/(1+(math.e ** -(x)))
    except OverflowError:
        if x > 0:
            return 1
        else:
            return 0

def load_dataset(ds_path, training=True, mnist_format=True):
    if mnist_format:
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
    Network("./dataset")

if __name__ == "__main__":
    main()
