# Handwriting Recognition

### About
This project uses neural networks to solve the problem of handwritten digit
recognition. It uses the MNIST database (http://yann.lecun.com/exdb/mnist/) for
training and testing


### Dependencies
This dependencies for this program are all contained in ./Pipfile. To install it, just run
'pipenv sync'. (Note that this does require pipenv to be installed, which can be done with
'pip install pipenv')

### Structure
This structure of the network used in this neural network is:
- 784 Input Neurons
- 2 Hidden Layers
  - 10 Neurons per layer
- 10 Output Neurons

### Usage
When the program is started, it will automatically begin training (Unless -t is included). To switch
between testing and training, simply press 't'.

##### Flags
- -d <dataset>: Specify custom dataset to use
- -j: Alert the program that the data is in JSON format. This is required if the data is JSON.
- -w <weights>: Specify custom, pregenerated weights to use
- -b <biases>: Specify custom, pregenerated weights to use
- -s <batch size>: Specify custom batch size to use (Default 100)
- -a <learning rate>: Specify learning rate (>0, Default 0.01)
- -t: Skip training and immediately begin testing

##### Mid-Usage Keys
- t: Switch between testing and training
- d: Toggle visibility of outputs (Disable for quicker learning, enable to see what's happening)

### Datasets
The data fed into this network should either be from the MNIST dataset or in JSON format.
For examples of what these should look like, check the dataset folder of this repository.
