# Handwriting Recognition

### About
This project uses neural networks to solve the problem of handwritten digit
recognition. It uses the MNIST database (http://yann.lecun.com/exdb/mnist/) for
training and testing

### Structure
This structure of the network used in this neural network is:
- 784 Input Neurons
- 3 Hidden Layers
  - 8 Neurons per layer
- 10 Output Neurons

### Usage
The program can be run from the command line or by clicking on it, however
to specify a custom dataset or to switch it into test mode it must be run from
the commmand line to allow the usage of flags

##### Flags
- -t, --test: Switch into test mode
- -d \<dataset>, --dataset <dataset>: Specify custom dataset to use