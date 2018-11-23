#!/usr/bin/env python3

import sys
import math
import random

from mnist import MNIST #Reading of datasets

class Network():
    def __init__(self, ds, mbs=100, seed=1):
        self.ds = ds
        self.seed = seed
        self.mbs = mbs
        self.dataset = load_dataset(ds) #Pull dataset, seperate for efficiency
        self.min_batch_size = mbs
        self.img_sets = []
        self.correct = 0
        self.num_guesses = 0

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


        self.layers.append(Layer([], True)) #Create output layer

        for i in range(10): #Output layer generation
            new_neuron = Neuron([])
            self.neurons.append(new_neuron)
            self.layers[-1].neurons.append(new_neuron)
            for neuron in self.layers[-2].neurons: #Add connection to all neurons from last layer
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

    def cost(self, expected):
        cost = 0
        for neuron, expected in zip(self.layers[-1].neurons, expected):
            cost += (neuron.output - expected) ** 2

        self.cost = cost

    def backprop(self):
        train_set = random.choice(self.img_sets)

        for train in train_set:
            self.activate(train)

            #Accuracy Scoring
            self.num_guesses += 1
            if self.output == train_set[train]:
                self.correct += 1

            outputs = []
            for neuron in self.layers[-1].neurons:
                outputs.append(str(round(neuron.output, 4)) + ('0' * (6 -len(str(round(neuron.output, 4))))))

            print(
                "Network Outputs:", outputs,
                "Network Output:", self.output,
                "Answer:", train_set[train],
                "Accuracy:",self.correct / self.num_guesses * 100,
                "(", self.correct,'/',self.num_guesses,')'
                )

            for layer in self.layers[::-1]:
                for neuron in layer.neurons:
                    if not hasattr(neuron, 'expected'):
                        neuron.expected = 0.5

                    if neuron in self.layers[-1].neurons:
                        neuron.expected = 0

                    if self.neurons.index(neuron) == len(self.neurons) - train_set[train]:
                        neuron.expected = 1

                    # Bias update
                    neuron.bias += neuron.expected - neuron.output #Update bias - Averaging later

                    # Connection Update
                    if neuron.output > neuron.expected:
                        for connection in neuron.inputs:
                            connection.neurons[0].expected = neuron.output - neuron.expected
                            if connection.weight > 0:
                                connection.weight -= neuron.output - neuron.expected
                            else:
                                connection.weight += neuron.output - neuron.expected

                    elif neuron.output < neuron.expected:
                        for connection in neuron.inputs:
                            connection.neurons[0].expected = neuron.expected - neuron.output
                            if connection.weight > 0:
                                connection.weight += neuron.expected - neuron.output
                            else:
                                connection.weight -= neuron.expected - neuron.output

                    delattr(neuron, 'expected')
                    #Clean up expected attr - Never used from earlier layers

        for neuron in self.neurons:
            neuron.bias /= len(list(train_set.keys())[0]) #Average neuron bias
            neuron.bias = sigmoid(neuron.bias)

    def test(self):
        self.dataset = load_dataset(self.ds, False) #Pull dataset, seperate for efficiency
        self.img_sets = []
        self.correct = 0
        self.num_guesses = 0

        for ds in self.dataset: #Randomly shuffle datasets
            random.seed(self.seed) #Seed to ensure images and values stay same
            random.shuffle(ds)

        curr_set = {} #Generate miniset
        for img, val in zip(self.dataset[0], self.dataset[1]):
            if len(curr_set) % self.min_batch_size == 0 and len(curr_set):
                self.img_sets.append(curr_set) #Add current set to image sets
                curr_set = {}
            else:
                curr_set[tuple(img)] = val

        self.img_sets.append(curr_set) #Add current set as otherwise never added

        while True:
            img_set = random.choice(self.img_sets)
            for img in img_set:
                self.activate(img)
                self.num_guesses += 1
                if self.output == img_set[img]:
                    self.correct += 1

                outputs = []
                for neuron in self.layers[-1].neurons:
                    outputs.append(str(round(neuron.output, 4)) + ('0' * (6 -len(str(round(neuron.output, 4))))))

                print(
                    "Network Outputs:", outputs,
                    "Network Output:", self.output,
                    "Answer:", img_set[img],
                    "Accuracy:",self.correct / self.num_guesses * 100,
                    "(", self.correct,'/',self.num_guesses,')'
                    )


class Layer():
    def __init__(self, neurons, out=False):
        self.neurons = neurons
        self.out = out

class Connection():
    def __init__(self, neurons):
        self.neurons = neurons
        self.weight = random.uniform(-1,1)

class Neuron():

    def __init__(self, inputs, input_neuron=None):
        self.input_neuron = input_neuron
        self.output = 0
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
        return 1/(1+(math.e ** -(x)))
    except OverflowError:
        if x > 0:
            return 1
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
    dataset = 'dataset'
    for arg in sys.argv:
        if '-d' in arg:
            try:
                dataset = sys.argv[sys.argv.index(arg)+1]
            except:
                raise IndexError("Dataset must be supplied after", arg, "flag")

    network = Network(dataset)

    while True:
        try:
            network.backprop()
        except KeyboardInterrupt:
            print('\n\n\n')
            print("TESTING")
            print('\n\n\n')
            network.test() #Loop inside func

if __name__ == "__main__":
    main()