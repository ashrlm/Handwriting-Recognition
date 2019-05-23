#!/usr/bin/env python3

import ast
import math
import sys
import json

import numpy as np
import pyxhook

try:
    from mnist.loader import MNIST #Reading of datasets
except ImportError:
    print("MNIST could not be imported. Depending on the format of the dataset, this may not be a problem")

class Network:

    def __init__(self, ds_path, mnist_format, weights, biases, batch_size, learning_rate, testing):
        #Initialise weights
        if weights:
            npz_weights = np.load(weights)
            self.weights = [
                npz_weights[npz_weights.files[0]],
                npz_weights[npz_weights.files[1]],
                npz_weights[npz_weights.files[2]]
            ]
        else:
            self.weights = [
                np.random.randn(10, 784), #Input -> h1
                np.random.randn(10, 10),  #h1    -> h2
                np.random.randn(10, 10)   #h2    -> Output
            ]

        if biases:
            self.biases = np.load(biases)[0]
        else:
            self.biases = np.random.uniform(-1, 1, 30)

        self.learning_rate = learning_rate

        #Load datasets
        raw_datasets  = load_dataset(ds_path, True, mnist_format)
        self.sets, self.labels  = raw_datasets[0], raw_datasets[1]
        self.batches = [] #[{item1: label1, item_2: label_2}, {item_101: label_101}]
        for i in range(0, math.floor(len(self.sets) / batch_size)):
            batch = {}
            for j in range(batch_size):
                batch[tuple(self.sets[j])] = self.labels[j]
            self.batches.append(batch)

        #Setup hooks
        self.hookmanager = pyxhook.HookManager()
        self.hookmanager.KeyDown = self.kbevent
        self.hookmanager.HookKeyboard()
        self.hookmanager.start()
        self.prev_ascii = None
        self.shown = True
        self.testing = testing
        self.running = True

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

    def kbevent(self, event):
        if not self.running:
            return
        if event.Ascii == 100:
            self.shown = not self.shown
        elif event.Ascii == 116:
            print("Training:", self.testing)
            self.testing = not self.testing
        self.prev_ascii = event.Ascii

    def train(self):
        pass

    def test(self):
        #Misc info for user
        total_attempts   = 0
        correct_attempts = 0
        error            = 0

        while self.testing and self.running:
            sample_index = np.random.randint(len(self.sets))
            test, label = self.sets[sample_index], self.labels[sample_index]
            final_activations = self.activate(test)
            res_index =  final_activations.index(max(final_activations))

            total_attempts += 1
            if res_index == label:
                correct_attempts += 1

            accuracy = 100 * (correct_attempts / total_attempts)

            if self.shown:
                for i in range(10):
                    if res_index == i:
                        error += (1-final_activations[i])**2
                    else:
                        error += (-final_activations[i])**2
                error /= 10

                print("Output:", res_index, "Correct answer:", label, "Accuracy:", str(accuracy)[:10]+"0"*(10-len(str(accuracy)[:10])), "LL Error:", str(error*100)[:10]+"%")

    def run(self):
        while True:
            if not self.testing:
                self.train()
            else:
                self.test()

def sigmoid(x):
    if x < 0:
        return 1 - 1 / (1 + math.exp(x))
    return 1 / (1 + math.exp(-x))

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

def parse():
    data = ["./dataset", True, None, None, 100, .01, False] #[ds_path, mnist_format, weights, biases, batch_size, learning rate, testing]

    if "-d" in sys.argv:
        data[0] = sys.argv[sys.argv.index("-d")+1]

    if "-j" in sys.argv:
        data[1] = False

    if "-w" in sys.argv:
        data[2] = sys.argv[sys.argv.index("-w")+1]

    if "-b" in sys.argv:
        data[3] = sys.argv[sys.argv.index("-b")+1]

    if "-s" in sys.argv:
        data[4] = int(sys.argv[sys.argv.index("-s")+1])

    if "-a" in sys.argv:
        data[5] = abs(float(sys.argv[sys.argv.index("-a")+1]))

    if "-t" in sys.argv:
        data[5] = True

    return data

def main():
    network = Network(*parse())
    try:
        network.run()
    except KeyboardInterrupt:
        network.running = False
        if (input("Save weights? [Y/n] ")+" ").lower()[0] != "n":
            np.savez('./weights', *network.weights)
        if (input("Save biases? [Y/n] ")+" ").lower()[0] != "n":
            np.save('./biases.npy', [network.biases]) #Wrap in new array to prevent 0d arrays

        network.hookmanager.cancel()

if __name__ == "__main__":
    main()
