import numpy as np
from matplotlib import pyplot as plt
from NeuralNetwork import NeuralNetwork

sigmoid = lambda x: 1/(1+np.exp(-x))
dsigmoid = lambda x: np.exp(-x)/np.power(np.exp(-x)+1, 2)

n = NeuralNetwork([2, 2, 2], 0.01, [sigmoid, sigmoid], [dsigmoid, dsigmoid])
t = np.random.random(size=(1, 2))
tr = np.random.random(size=(1, 2))
r = n.run(t)

print(t, " ->", r, "(test aleatoire)")

x, h, e = n.train(t,tr)
print("finished")

