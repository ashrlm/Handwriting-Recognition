#!/usr/bin/env python3

import math
import random

from mnist import MNIST #Reading of datasets

class Network():
    def __init__(self, ds, mbs=100, seed=1):
        self.dataset = load_dataset(ds, False) #Pull dataset, seperate for efficiency
        self.min_batch_size = mbs
        self.img_sets = []

        for ds in self.dataset: #Randomly shuffle datasets
            random.seed(seed) #Seed to ensure images and values stay same
            random.shuffle(ds)

        curr_set = {} #Generate miniset
        for img, val in zip(self.dataset[0], self.dataset[1]):
            if len(curr_set) % self.min_batch_size == 0 and len(curr_set):
                self.img_sets.append(curr_set) #Add current set to image sets
                curr_set = {}
            else:
                curr_set[tuple(img)] = val

        self.img_sets.append(curr_set) #Add current set as otherwise never added

        self.layers = [Layer([])]
        self.neurons = []
        self.connections = []
        self.input_neurons = []

        for i in range(784): #Input layer generation
            input_neuron = Neuron([], input_neuron=True)
            self.neurons.append(input_neuron)
            self.layers[0].neurons.append(input_neuron)
            self.input_neurons.append(input_neuron)

        for i in range(2): #Hidden layer generation
            self.layers.append(Layer([]))
            for i in range(10):
                new_neuron = Neuron([])
                self.neurons.append(new_neuron)
                self.layers[-1].neurons.append(new_neuron)
                for neuron in self.layers[-2].neurons: #Add connections to all neurons from last layer
                    new_conn = Connection(
                        (neuron, new_neuron)
                    )
                    new_neuron.inputs.append(new_conn)
                    self.connections.append(new_conn)

        for i in range(10): #Output layer generation
            new_neuron = Neuron([])
            self.neurons.append(new_neuron)
            for neuron in self.layers[-1].neurons: #Add connection to all neurons from last layer
                new_conn = Connection(
                    (neuron, new_neuron)
                )
                new_neuron.inputs.append(new_conn)
                self.connections.append(new_conn)

    def activate(self, img):

        for px_val, neuron in zip(img, self.layers[0].neurons): #Set output on input neurons
            neuron.output = px_val

        for layer in self.layers[1:]:
            for neuron in layer.neurons:
                neuron.activate()

        # Get neuron with highest fire rate and set that as decision to prove learning


        max_neuron_index = -1
        max_neuron_output = -1

        for i, neuron in enumerate(self.layers[-1].neurons):
            if neuron.output > max_neuron_output:
                max_neuron_index = i
                max_neuron_output = neuron.output

        self.output = max_neuron_index

    def backprop(self):
        train_set = random.choice(self.img_sets)
        weight_bias_changes = []

        for train in train_set:
            expected = [0] * 10
            expected[train_set[train]] = 1
            self.activate(train)

            # TODO: Add code here to check weight_bias_changes for gradient descent

class Layer():
    def __init__(self, neurons):
        self.neurons = neurons

class Connection():
    def __init__(self, neurons):
        self.neurons = neurons
        self.weight = random.uniform(-1,1)

class Neuron():

    def __init__(self, inputs, input_neuron=None):
        self.input_neuron = input_neuron
        self.inputs = inputs #List of connection objects
        self.bias = random.uniform(-1, 1) #Generate bias

    def activate(self):
        if not self.input_neuron:
            weighted_input = self.bias
            for input in self.inputs:
                weighted_input += input.neurons[0].output * input.weight
            self.output = sigmoid(weighted_input)

def sigmoid(x):
    try:
        return 1/(1+(math.e ** -x))
    except OverflowError:
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

def load_dataset(ds_path, training=True):
    # NOTE: Returns (Dataset images, Dataset labels)
    dataset = MNIST(ds_path)
    if training:
        return (dataset.load_training()[0], dataset.train_labels)
    else:
        return (dataset.load_testing()[0], dataset.test_labels)

def main():
    pass

if __name__ == "__main__":
    main()