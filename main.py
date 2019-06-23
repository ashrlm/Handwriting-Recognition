#!/usr/bin/env python3

import ast
import math
import sys
import time
import json
import threading

import numpy as np
import pynput

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

        #Load training datasets
        raw_datasets  = load_dataset(ds_path, True, mnist_format)
        self.sets, self.labels  = raw_datasets[0], raw_datasets[1]
        self.batches = [] #[{item1: label1, item_2: label_2}, {item_101: label_101}]
        for i in range(0, math.floor(len(self.sets) / batch_size)):
            batch = {}
            for j in range(batch_size):
                batch[tuple(self.sets[j])] = self.labels[j]
            self.batches.append(batch)

        raw_datasets_test = load_dataset(ds_path, False, mnist_format)
        self.test_sets, self.test_labels = raw_datasets_test[0], raw_datasets_test[1]
        #Batches not needed for testing

        #Setup hooks
        def kb_start():
            with pynput.keyboard.Listener(on_press = self.kb_press, on_release = self.kb_release) as listener:
                listener.join()

        self.kb_thread = threading.Thread(target=kb_start)
        self.kb_thread.daemon = True
        self.kb_thread.start()

        self.prev_key = None
        self.shown = True
        self.testing = testing
        self.running = True
        self.mode_delay = False

    def activate(self, sample):
        activs = [[0] * 784]
        for i, data in zip(range(784), sample): #Activate input layer
            activs[0][i] = data

        for i in range(3):
            activations_curr = []
            for neuron in range(10):
                activation = 0
                for old_activ, weight in zip(activs[-1], self.weights[i][neuron]):
                    activation += (old_activ * weight) + self.biases[(10*i) + neuron]

                activations_curr.append(activation)

            activs.append(activations_curr)

        return activs[1:]

    def kb_press(self, key):
        pass

    def kb_release(self, key):
        if not self.running:
            return
        if key == pynput.keyboard.KeyCode.from_char('d'):
            self.shown = not self.shown
        elif key == pynput.keyboard.KeyCode.from_char('t'):
            print("Training:", self.testing)
            self.testing = not self.testing
        self.prev_key = str(key)
        self.mode_delay = True

    def train(self):

        #NOTES
        #Activs = [[L1_Activs], [L2_Activs], [L3_Activs], [L4_ACTIVS]]
        #Expectations = [L(N) Expectations]

        batch = self.batches[np.random.randint(0, len(self.batches))]
        expected = [label for label in list(batch.values())]
        Δws = [] #List of list of all weight derivates over all training examples
        Δbs = []
        for item, label in zip(batch, expected):
            activs = self.activate(item)
            res_index = activs[-1].index(max(activs[-1]))

            #Misc info for user
            self.train_attempts += 1
            if res_index == label:
                self.train_correct += 1

            accuracy = 100 * (self.train_correct / self.train_attempts)

            if self.shown:
                error = sum([(1-sigmoid(activ))**2 if res_index == label else sigmoid(activ)**2 for activ in activs[-1]]) / len(activs[-1])
                print("Output:", res_index, "Correct answer:", label, "Accuracy:", str(accuracy)[:10]+"0"*(10-len(str(accuracy)[:10])), "LL Error:", str(error*100)[:10]+"%")
            for i, layer in enumerate(activs[::-1]):
                Δw_sample = []
                Δb_sample = []
                for j, neuron in enumerate(layer):
                    Δb = sigmoid(neuron) * ((sigmoid(neuron)*(1-sigmoid(neuron)))) * (2*(sigmoid(neuron) - label)) #d(sigmoid)/d(w(L)(jk)) * (2(sigmoid(a(l)(jk))) - y(j))
                    Δb_sample.append(Δb)
                    Δa = 0
                    try:
                        for neuron_j in range(len(activs[::-1][i-1])):
                            Δw_sample.append(sigmoid(activs[::-1][i-1][neuron_j]) * Δb)
                            #Compute Δa(neuron)(L-1) here
                            try:
                                Δa += (self.weights[::-1][i][j][neuron_j]) * (sigmoid(activs[j][neuron_j])*(1-sigmoid(activs[j][neuron_j]))) * (2*(sigmoid(activs[j][neuron_j]) - expected[neuron_j]))
                                expected[neuron_j] += int(-self.learning_rate * Δa)
                            except IndexError: #Ensure sufficient weights
                                continue

                    except IndexError as e: #Handle last layer
                        pass

            Δws.append(Δw_sample)
            Δbs.append(Δb_sample)

        #Update biases
        for i in range(len(Δbs[0])):
            self.biases[::-1][i] += -self.learning_rate * (sum([biases[i] for biases in Δbs]) / len(Δbs))

        #Update weights
        weight_array = np.concatenate((self.weights[0].flatten(), self.weights[1].flatten(), self.weights[2].flatten())).flatten()[::-1]
        for i in range(len(Δws[0])):
            weight_array[::-1][i] = weight_array[i] + (-self.learning_rate * (sum([weights[i] for weights in Δws]) / len(Δws)))

        self.weights[0] = weight_array[:7840].reshape(10, 784) #784 * 10
        self.weights[1] = weight_array[7840:7940].reshape(10, 10) #10*10 + 7840
        self.weights[2] = weight_array[7940:].reshape(10, 10) #Final 100 elements


    def test(self):
        sample_index = np.random.randint(len(self.test_sets))
        test, label = self.test_sets[sample_index], self.test_labels[sample_index]
        final_activations = [[sigmoid(a_l) for a_l in activation] for activation in self.activate(test)][-1]
        res_index =  final_activations.index(max(final_activations))

        #Misc info for user
        self.test_attempts += 1
        if res_index == label:
            self.test_correct += 1

        accuracy = 100 * (self.test_correct / self.test_attempts)

        if self.shown:
            error = sum([(1-activ)**2 if res_index == i else activ**2 for i, activ in enumerate(final_activations)]) / len(final_activations)
            print("Output:", res_index, "Correct answer:", label, "Accuracy:", str(accuracy)[:10]+"0"*(10-len(str(accuracy)[:10])), "LL Error:", str(error*100)[:10]+"%")

    def run(self):
        print("Training: True")

        self.train_attempts = 0
        self.train_correct = 0

        self.test_attempts = 0
        self.test_correct = 0

        while self.running:
            if self.mode_delay:
                self.mode_delay = False
                time.sleep(3)

            if self.testing:
                self.test()
            else:
                self.train()

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def inv_sig(x):
    return -log((1/x)-1, math.e)

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

if __name__ == "__main__":
    main()
