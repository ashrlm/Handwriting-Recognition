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
The program can be run from the command line or by clicking on it, however
to specify a custom dataset it must be run from the commmand line to allow the
usage of flags

When the program is ran, it will automatically begin training. To switch it from
training mode into testing mode, simply press CTRL-C.

##### Flags
- -d \<dataset>, --dataset <dataset>: Specify custom dataset to use