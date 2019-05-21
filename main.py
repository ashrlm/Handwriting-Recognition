#!/usr/bin/env python3

import ast
import math
import sys
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
                np.random.randn(10, 784), #Input -> h1
                np.random.randn(10, 10),  #h1    -> h2
                np.random.randn(10, 10)   #h2    -> Output
            ]

        self.biases = np.random.uniform(-1, 1, 30)

        #Load datasets
        raw_datasets  = load_dataset(ds_path, True, mnist_format)
        self.sets, self.labels  = raw_datasets[0], raw_datasets[1]
        self.batches = [] #[{item1: label1, item_2: label_2}, {item_101: label_101}]
        for i in range(0, math.floor(len(self.sets) / batch_size)):
            batch = {}
            for j in range(batch_size):
                batch[tuple(self.sets[j])] = self.labels[j]
            self.batches.append(batch)

    def activate(self, sample):
        activations_prior = [0] * 784 #Store previous layer activations
        for i, data in zip(range(784), sample): #Activate input layer
            activations_prior[i] = data

        for i in range(3):
            activations_curr = []
            for neuron in range(10):
                activation = 0
                for old_activ, weight in zip(activations_prior, self.weights[i][neuron]):
                    activation += (old_activ * weight) + self.biases[(10*i) + neuron]

                activations_curr.append(sigmoid(activation))

            activations_prior = activations_curr

        return activations_prior

    def test(self):
        #Misc info for opertator
        total_attempts   = 0
        correct_attempts = 0
        error            = 0

        while True:
            sample_index = np.random.randint(len(self.sets))
            test, label = self.sets[sample_index], self.labels[sample_index]
            final_activations = self.activate(test)
            res_index =  final_activations.index(max(final_activations))

            total_attempts += 1
            if res_index == label:
                correct_attempts += 1

            for i in range(10):
                if res_index == i:
                    error += (1-final_activations[i])**2
                else:
                    error += (-final_activations[i])**2
            error /= 10

            accuracy = 100 * (correct_attempts / total_attempts)
            print("Output:", res_index, "Correct answer:", label, "Accuracy:", str(accuracy)[:10], "LL Error:", str(error)[:10])

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

def parse():
    data = ["./dataset", True, None, 100, False] #[ds_path, mnist_format, weights, batch_size, testing]

    if "-d" in sys.argv:
        data[0] = sys.argv[sys.argv.index("-d")+1]

    if "-j" in sys.argv:
        data[1] = False

    if "-w" in sys.argv:
        data[2] = sys.argv[sys.argv.index("-w")+1]

    if "-b" in sys.argv:
        data[3] = sys.argv[sys.argv.index("-b")+1]

    if "-t" in sys.argv:
        data[4] = True

    return data

def main():
    network = Network(*parse()[:-1])
    try:
        network.test()
    except KeyboardInterrupt:
        if input("Save weights? [Y/n]").lower() != "n":
            save_weights('./weights.txt', network.weights)

if __name__ == "__main__":
    main()
