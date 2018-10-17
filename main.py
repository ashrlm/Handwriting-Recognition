import math

import sympy #Calculation of partial derivatives
from mnist import MNIST #Reading of datasets

class Network():
    def __init__(self):
        self.layers = [Layer()]
        self.neurons = []
        self.connections = []
        self.input_neurons = []

        for i in range(784): #Input layer generation
            input_neuron = Neuron([])
            self.neurons.append(input_neuron)
            self.layers[0].neurons.append(input_neuron)
            self.input_neurons.append(input_neuron)

        for i in range(2): #Hidden layer generation
            self.layers.append(Layer([]))
            for i in range(10):
                new_neuron = Neuron()
                self.neurons.append(new_neuron)
                for neuron in self.layers[-1].neurons:
                    new_conn = Connection(
                        (neuron, new_neuron)
                    )

                    new_neuron.inputs.append(new_conn)
                    self.connections.append(new_conn)

        for i in range(10): #Output layer generation
            pass



    def activate(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.activate()

    def cost(self):
        pass

    def part_deriv(self):
        pass

    def gradient_descent(self):
        pass

class Layer():
    def __init__(self, neurons):
        self.neurons = neurons

class Connection():
    def __init__(self, neurons, weight):
        self.neurons = neurons
        self.weight = weight

class Neuron():
    def __init__(self, inputs, output=None):
        if output:
            self.output = output
            self.input_neuron = True
        self.inputs = inputs #List of connection objects

    def activate(self):
        if not self.input_neuron:
            weighted_input = 0
            for input in self.inputs:
                weighted_input += input.output * input.weight
            self.output = sigmoid(weighted_input)

def sigmoid(x):
    return 1/(1+(math.e ** -x))

def load_dataset(ds_path, training=True):
    dataset = MNIST(ds_path)
    if training:
        return dataset.load_training()
    else:
        return dataset.load_testing()

def main():
    Network()

if __name__ == "__main__":
    main()