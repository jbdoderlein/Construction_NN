import mnist
import cupy as cp
import numpy as np
from matplotlib import pyplot as plt
from CupyNeuralNetwork import NeuralNetwork
import time


def convert_label(labels):
    newlab = [0] * len(labels)
    for i in range(len(labels)):
        t = cp.zeros(10)
        t[labels[i]] = 1
        newlab[i] = t[:, cp.newaxis]
    return cp.array(newlab)


def max_output(v):
    maximum, imax = 0, 0
    for i in range(len(v)):
        if v[i] > maximum:
            maximum, imax = v[i], i
    return imax


x_train, t_train, x_test, t_test = mnist.load()
x_train, t_train, x_test, t_test = cp.array(x_train), cp.array(t_train), cp.array(x_test), cp.array(t_test)

tjb_train = convert_label(t_train)
tjb_test = convert_label(t_test)


def img_show(array):
    pixels = array.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()


sigmoid = lambda x: 1 / (1 + cp.exp(-x))
dsigmoid = lambda x: x * (1 - x)
tanh = cp.tanh
dtanh = lambda x: 1 - (cp.tanh(x) ** 2)

if __name__ == '__main__':
    T1 = time.time()
    n = NeuralNetwork([784, 950, 500, 150, 10], 0.01, [tanh, sigmoid, tanh, sigmoid], [dtanh, dsigmoid, dtanh, dsigmoid])

    BATCH = 100
    BATCH_SIZE = 10
    losses = cp.zeros(BATCH)
    losses2 = cp.zeros(BATCH)

    for i in range(BATCH):
        t1 = time.time()
        for j in range(BATCH_SIZE):
            rn = cp.random.randint(59999)
            n.train([x_train[rn]], [tjb_train[rn]])
        t2 = time.time()
        u = n.losses(x_test[:10], tjb_test[:10])
        losses[i] = u
        t3 = time.time()
        print(f"Batch : {i + 1}/{BATCH} en {round(t2 - t1, 4)} s avec {round(t3 - t2, 4)} en loss")
    T2 = time.time()
    print("Le tout en ", T2-T1)
    #n.save("mninst_data_1")
    """
    plt.figure()
    #plt.plot(losses)
    plt.plot(losses2)
    plt.xlabel("BATCH")
    plt.ylabel("Loss")
    plt.title(f"MNIST Loss with {BATCH} batchs of {BATCH_SIZE} retropopagation")
    plt.show()
"""