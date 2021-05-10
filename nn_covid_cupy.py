import numpy as np
import cupy as cp
from matplotlib import pyplot as plt
from CupyNeuralNetwork import NeuralNetwork
import time
from os import listdir
from os.path import isfile, join
from PIL import Image
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="jb_dataset_covid")
ap.add_argument("-s", "--size", type=int, default=30)
ap.add_argument("-tp", "--test_proportion", type=float, default=0.2)
args = vars(ap.parse_args())

## Dataset Init

SIZE = args["size"]  # Taille des images
test_proportion = args["test_proportion"]  # Proportion des images classé comme test
dataset_name = args["dataset"]

img_train = []
label_train = []
img_test = []
label_test = []

covidfiles = [f for f in listdir(f'{dataset_name}/covid') if isfile(join(f'{dataset_name}/covid', f))]
normalfiles = [f for f in listdir(f'{dataset_name}/normal') if isfile(join(f'{dataset_name}/normal', f))]

covidfiles_size = len(covidfiles)
normalfiles_size = len(normalfiles)
print(f"{covidfiles_size} covid patient and {normalfiles_size} non-covid patient")

for i in range(covidfiles_size):
    try:
        image = Image.open(f'{dataset_name}/covid/{covidfiles[i]}').convert('L')
        image_array = cp.array(image.resize((SIZE, SIZE))).reshape((SIZE ** 2,))
        label = cp.array([0, 1])
        if i < covidfiles_size * (1 - test_proportion):  # Image train
            img_train.append(image_array)
            label_train.append(label[:, cp.newaxis])
        else:  # Image test
            img_test.append(image_array)
            label_test.append(label[:, cp.newaxis])
    except Exception as e:
        print("error with ", i)

for i in range(normalfiles_size):
    try:
        image = Image.open(f'{dataset_name}/normal/{normalfiles[i]}').convert('L')
        image_array = cp.array(image.resize((SIZE, SIZE))).reshape((SIZE ** 2,))
        label = cp.array([0, 1])
        if i < normalfiles_size * (1 - test_proportion):  # Image train
            img_train.append(image_array)
            label_train.append(label[:, cp.newaxis])
        else:  # Image test
            img_test.append(image_array)
            label_test.append(label[:, cp.newaxis])
    except Exception as e:
        print("error with ", i)

# NN Init


# Fonctions d'activations
sigmoid = lambda x: 1 / (1 + cp.exp(-x))
dsigmoid = lambda x: x * (1 - x)
tanh = cp.tanh
dtanh = lambda x: 1 - (cp.tanh(x) ** 2)
relu = lambda x: cp.maximum(x, 0)
drelu = lambda x: cp.where(x > 0, 1, 0)
elu = lambda x: cp.where(x >= 0, x, cp.exp(x) - 1)
delu = lambda x: cp.where(x > 0, 1, cp.exp(x))

n = NeuralNetwork([SIZE ** 2, 3000, 2000, 500, 50, 2], 0.01, [tanh, sigmoid, tanh, tanh, sigmoid],
                  [dtanh, dsigmoid, dtanh, dtanh, dsigmoid])

BATCH = 20  # Nombre de batch
EPOCH = 10  # Nombre di'mage avant retropopagation
BATCH_SIZE = 5  # Nombre d'epoch (et donc entre chaque calcul de loss)
losses = cp.zeros(BATCH)

## NN Execution

for i in range(BATCH):
    t1 = time.time()
    for j in range(BATCH_SIZE):
        img, label = [], []
        for t in range(EPOCH):
            rn = int(cp.random.randint(len(img_train) - 1))
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
plt.plot(losses.get())
plt.xlabel("BATCH")
plt.ylabel("Loss")
plt.title(f"Covid Loss with {BATCH} batchs of {BATCH_SIZE} retropopagation")
plt.savefig('plot/test.png')
