import network
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot= True)
x = mnist.train.images
y = mnist.train.labels
training_data = []
test_data = []
for i, j in zip(x[:45000], y[:45000]) :
    training_data.append((i,j))

for i, j in zip(x[45000:], y[45000:]) :
    test_data.append((i, j))

net = network.Network([784,30,10])
net.SGD(training_data,30, 30, 0.001, test_data = test_data)
