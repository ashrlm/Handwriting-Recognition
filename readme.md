# Handwriting Recognition

### About
This project uses neural networks to solve the problem of handwritten digit
recognition. It uses the MNIST database (http://yann.lecun.com/exdb/mnist/) for
training and testing


### Dependencies
This program requires the `python-mnist` package from PyPI.
This can be installed by running `python -m pip install python-mnist`
from the command line

### Structure
This structure of the network used in this neural network is:
- 784 Input Neurons
- 2 Hidden Layers
  - 10 Neurons per layer
- 10 Output Neurons

### Usage
When the program is started, it will automatically begin training (Unless -t is included) To switch it from
training mode into testing mode, simply press CTRL-C.

##### Flags
- -d <dataset>: Specify custom dataset to use
- -j: Alert the program that the data is in JSON format. This is required if the data is JSON.
- -w <weights>: Specify custom, pregenerated weights to use
- -b <biases>: Specify custom, pregenerated weights to use
- -s <batch size>: Specify custom batch size to use (Default 100)
- -t: Skip training and immediately begin testing

### Datasets
The data fed into this network should either be from the MNIST dataset or in JSON format.
For examples of what these should look like, check the dataset folder of this repository.