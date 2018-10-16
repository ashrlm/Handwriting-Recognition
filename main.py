#!usr/bin/env/python3

import math

import sympy #Calculation of partial derivatives
from mnist import MNIST #Reading of datasets

class Network():
    def __init__(self,layers,  neurons, connections):
        self.layers = layers
        self.neurons = neurons
        self.connections = connections

    def activate(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.activate()

class Layer():
    def __init__(self, neurons):
        self.neurons = neurons

class Connection():
    def __init__(self, neurons, weight):
        self.neurons = neurons
        self.weight = weight

class Neuron():
    def __init__(self, inputs, output=None):
        if self.output:
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