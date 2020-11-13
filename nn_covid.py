import numpy as np
from matplotlib import pyplot as plt
from NeuralNetwork import NeuralNetwork
import time
from PIL import Image


sigmoid = lambda x: 1 / (1 + np.exp(-x))
dsigmoid = lambda x: x * (1 - x)
tanh = np.tanh
dtanh = lambda x: 1 - (np.tanh(x) ** 2)

n = NeuralNetwork([10000, 5000, 500, 150, 5], 0.01, [tanh, sigmoid, tanh, sigmoid], [dtanh, dsigmoid, dtanh, dsigmoid])

image = Image.open('example.jpeg')
image_test = np.array(image.resize((100,100))).reshape((10000,))

n.forward_propagation(image_test)