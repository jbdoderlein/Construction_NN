import numpy as np
from matplotlib import pyplot as plt
from NeuralNetwork import NeuralNetwork

sigmoid = lambda x: 1 / (1 + np.exp(-x))
dsigmoid = lambda x: x * (1 - x)
tanh = np.tanh
dtanh = lambda x: 1 - (np.tanh(x)**2)

n = NeuralNetwork([2, 3, 1], 0.01, [tanh, sigmoid], [dtanh, dsigmoid])
training_data = np.array([
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0]
])

training_labels = np.array([
    [1],
    [1],
    [0],
    [0]
])

BATCH = 200
EPOCH_PER_BATCH = 50

print("$ Pre entrainement")
for i, j in zip(training_data, training_labels):
    print("With",i, " ->", n.forward_propagation(i)[0, 0], " (except",j[0],")")


losses = np.zeros(BATCH)
for i in range(BATCH):
    print("Epoch", i)
    n.train(training_data, training_labels, EPOCH_PER_BATCH)
    u = n.losses(training_data, training_labels)
    losses[i] = u


print("$ Post entrainement")
for i, j in zip(training_data, training_labels):
    print("With",i, " ->", n.forward_propagation(i)[0, 0], " (except",j[0],")")

t = np.linspace(1,BATCH,BATCH)
f = 1/np.sqrt(t)
plt.figure()
plt.plot(t, losses, "r")
plt.xlabel("BATCH")
plt.ylabel("Loss")
plt.show()