import network
import numpy as np
import pandas as pd

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot= True)

df = pd.DataFrame(mnist)
print(df.columns)
