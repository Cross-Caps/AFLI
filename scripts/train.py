# -*- coding: utf-8 -*-
"""
**Analysing FC networks with random weight matrices on  MNIST/Cifar-10 dataset**
"""

# Import required packages
import pickle
import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10


# Import training utilities
# train- for network with Gaussian weights
# trainO- for network with orthogonal weights

from utils_fixed_a import train, trainO        # When training with a fixed value of parameter 'a'
#from utils_variable_a import train, trainO    # When training with a adaptive variable value of parameter 'a'






# Loading and preprocessing MNIST/Fashion-MNIST/Cifar10

#dim = 784 # Image dimension MNIST/Fashion-MNIST
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#num_pixels = X_train.shape[1] * X_train.shape[2]

#(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
#num_pixels = X_train.shape[1] * X_train.shape[2]


dim = 3072 # Image dimension Cifar-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
num_pixels = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]


# Create Train/Test Splits
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
X_train = X_train / 255 
X_test = X_test / 255 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]


# Set training hyper-parameters
trials = 10
width = 300
nb_epochs = 100
batchsize = 128
depth = [20, 50, 100, 200]
Learning_alg = SGD(lr = .0001)


# Choose activation function
act='S_exp'

# Pre-computed values of (sigma_b, sigma_w, q*) corresponding to value of parameter 'a'
params = [[  1.00000000e+00 ,  1.32529293e+00 , 1.45256071e+00],
          [  1.00000000e-01 ,  1.04080355e+00 , 3.91394299e-01],
          [  1.00000000e-02 ,  1.00443422e+00 , 1.82999134e-01],
		  [  1.00000000e-03 ,  1.00050152e+00 , 1.14086494e-01],
          [  1.00000000e-04 ,  1.00009256e+00 , 8.69785199e-02]] 
 


# Main training loop
for L in depth:
  for param in params:
    var_b, var_w, q = param
    for trial in range(trials):
      train(X_train, X_test, y_train, y_test, dim, width, L, act, var_w, var_b, q, Learning_alg, nb_epochs, batchsize,trial)





