import mnist
import numpy as np
from matplotlib import pyplot as plt
from NeuralNetwork import NeuralNetwork
import time


def convert_label(labels):
    newlab = [0] * len(labels)
    for i in range(len(labels)):
        t = np.zeros(10)
        t[labels[i]] = 1
        newlab[i] = t[:, np.newaxis]
    return np.array(newlab)


def max_output(v):
    maximum, imax = 0, 0
    for i in range(len(v)):
        if v[i] > maximum:
            maximum, imax = v[i], i
    return imax


x_train, t_train, x_test, t_test = mnist.load()

tjb_train = convert_label(t_train)
tjb_test = convert_label(t_test)


def img_show(array):
    pixels = array.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()


sigmoid = lambda x: 1 / (1 + np.exp(-x))
dsigmoid = lambda x: x * (1 - x)
tanh = np.tanh
dtanh = lambda x: 1 - (np.tanh(x) ** 2)

n = NeuralNetwork([784, 950, 500, 150, 10], 0.01, [tanh, sigmoid, tanh, sigmoid], [dtanh, dsigmoid, dtanh, dsigmoid])

BATCH = 150
BATCH_SIZE = 400
losses = np.zeros(BATCH)

for i in range(BATCH):
    t1 = time.time()
    for j in range(BATCH_SIZE):
        rn = np.random.randint(59999)
        n.train([x_train[rn]], [tjb_train[rn]], 1)
    t2 = time.time()
    u = n.losses(x_test, tjb_test)
    losses[i] = u
    t3 = time.time()
    print(f"Batch : {i + 1}/{BATCH} en {round(t2 - t1, 4)} s avec {round(t3 - t2, 4)} en loss")

n.save("mninst_data_1")

plt.figure()
plt.plot(losses)
plt.xlabel("BATCH")
plt.ylabel("Loss")
plt.title(f"Loss with {BATCH} batchs of {BATCH_SIZE} retropopagation")
plt.show()
