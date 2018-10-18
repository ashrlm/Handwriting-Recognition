import math
import random

from mnist import MNIST #Reading of datasets

class Network():
    def __init__(self, ds, mbs=100, seed=1):
        self.dataset = load_dataset(ds, False) #Pull dataset, seperate for efficiency
        self.min_batch_size = mbs
        self.img_sets = {}

        for ds in self.dataset: #Randomly shuffle datasets
            random.seed(seed) #Seed to ensure images and values stay same
            random.shuffle(ds)

        for img, val in zip(self.dataset[0], self.dataset[1]):
            self.img_sets[tuple(img)] = val

        print(self.img_sets)


        self.layers = [Layer([])]
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

    def activate(self):

        # for img in self.dataset[0]:
#
            # for input_neuron, px_val in zip(self.layers[0].neurons, img):
                # input_neuron.output = px_val
                # print(type(px_val))
#
        # for layer in self.layers[1:]:
            # for neuron in layer.neurons:

        pass


    def cost(self):
        # dataset = load_dataset('dataset')
        # train_img = list(dataset[0][0])
        # train_labels = dataset[1]
#
        # costs = []
#
        # for img, label in zip(train_img, train_labels):
#
            # self.activate(img)
#
            # expected_out = [0] * 10
            # expected_out[label] = 1
#
            # for neuron, expected in zip(self.layers[-1].neurons, expected_out):
                # costs.append((neuron.output - expected) ** 2)

            pass


    def backprop(self):
        pass

class Layer():
    def __init__(self, neurons):
        self.neurons = neurons

class Connection():
    def __init__(self, neurons):
        self.neurons = neurons
        self.weight = random.uniform(-1,1)

class Neuron():
    def __init__(self, inputs, output=None):
        self.input_neuron = False
        if output:
            self.output = output
            self.input_neuron = True
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
    Network('dataset')

if __name__ == "__main__":
    main()