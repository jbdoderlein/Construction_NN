import numpy as np
import cupy as cp
from matplotlib import pyplot as plt
from CupyNeuralNetwork import NeuralNetwork
import time
from os import listdir
from os.path import isfile, join

## Dataset Init

SIZE = 100

img_train = []
label_train = []
img_test = []
label_test = []

covidfiles = [f for f in listdir('jb_dataset/covid') if isfile(join('compiled/covided', f))]
normalfiles = [f for f in listdir('jb_dataset/normal') if isfile(join('compiled/normal', f))]

test_proportion = 0.8
SIZE = 120

n = len(covidedfiles)
for i in range(n):
    print(i)
    image = Image.open(f'compiled/covided/{covidedfiles[i]}').convert('L')
    image_array = np.array(image.resize((SIZE, SIZE))).reshape((SIZE ** 2,))
    label = np.array([0, 1])
    if i < n*test_proportion:
        img_train.append(image_array)
        label_train.append(label[:, np.newaxis])
    else:
        img_test.append(image_array)
        label_test.append(label[:, np.newaxis])

n = len(normalfiles)
for i in range(n):
    print(i)
    image = Image.open(f'compiled/normal/{normalfiles[i]}').convert('L')
    image_array = np.array(image.resize((SIZE, SIZE))).reshape((SIZE ** 2,))
    label = np.array([1, 0])
    if i < n*test_proportion:
        img_train.append(image_array)
        label_train.append(label[:, np.newaxis])
    else:
        img_test.append(image_array)
        label_test.append(label[:, np.newaxis])


# NN Init


# Fonctions d'activations
sigmoid = lambda x: 1 / (1 + np.exp(-x))
dsigmoid = lambda x: x * (1 - x)
tanh = np.tanh
dtanh = lambda x: 1 - (np.tanh(x) ** 2)
relu = lambda x: np.maximum(x, 0)
drelu = lambda x: np.where(x > 0, 1, 0)
elu = lambda x: np.where(x >= 0, x, np.exp(x) - 1)
delu = lambda x: np.where(x > 0, 1, np.exp(x))

n = NeuralNetwork([14400, 15000, 2000, 500, 50, 2], 0.01, [tanh, sigmoid, tanh, tanh, sigmoid],
                  [dtanh, dsigmoid, dtanh, dtanh, dsigmoid])

BATCH = 100  # Nombre de batch
EPOCH = 20  # Nombre di'mage avant retropopagation
BATCH_SIZE = 10 # Nombre d'epoch (et donc entre chaque calcul de loss)
losses = np.zeros(BATCH)
losses2 = np.zeros(BATCH)

## NN Execution

for i in range(BATCH):
    t1 = time.time()
    for j in range(BATCH_SIZE):
        img, label = [], []
        for t in range(EPOCH):
            rn = cp.random.randint(len(img_train) - 1)
            img.append(img_train[rn])
            label.append(label_train[rn])
        n.train(img, label)
    t2 = time.time()
    u = n.losses(img_test, label_test)
    losses[i] = u
    t3 = time.time()
    print(f"Batch : {i + 1}/{BATCH} en {round(t2 - t1, 4)} s avec {round(t3 - t2, 4)} en loss ({EPOCH} {BATCH_SIZE})")


## NN representation

plt.figure()
plt.plot(losses)
plt.xlabel("BATCH")
plt.ylabel("Loss")
plt.title(f"Covid Loss with {BATCH} batchs of {BATCH_SIZE} retropopagation")
plt.imsave('test.png')
