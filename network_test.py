import network
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot= True)

t = input_data.read_data_sets("MNIST_data/", one_hot= False)
test_x = t.train.images[45000:]
test_x_reshaped = [i.reshape(784,1) for i in test_x]
test_y = t.train.labels[45000:]
x = mnist.train.images
y = mnist.train.labels

t_y = [i.reshape(10,1) for i in y]
t_x = [i.reshape(784,1) for i in x]

training_data = []
test_data = []

for i, j in zip(t_x, t_y) :
    training_data.append((i,j))

for i, j in zip(test_x_reshaped, test_y) :
    test_data.append((i, j))

net = network.Network([784,100,10])
net.SGD(training_data,300, 30, 3.0, test_data = test_data)

