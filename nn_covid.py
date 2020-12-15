import numpy as np
from matplotlib import pyplot as plt
from NeuralNetwork import NeuralNetwork
import time
import dill

with open('dataset/normal_covid_v2.dataset', 'rb') as f:
    covid = dill.load(f)

img_train, label_train, img_test, label_test = covid["img_train"], covid["label_train"], covid["img_test"], covid[
    "label_test"]

# Fonctiosn d'activations
sigmoid = lambda x: 1 / (1 + np.exp(-x))
dsigmoid = lambda x: x * (1 - x)
tanh = np.tanh
dtanh = lambda x: 1 - (np.tanh(x) ** 2)
relu = lambda x: np.maximum(x, 0)
drelu = lambda x: np.where(x > 0, 1, 0)
elu = lambda x: np.where(x >= 0, x, np.exp(x) - 1)
delu = lambda x: np.where(x > 0, 1, np.exp(x))

n = NeuralNetwork([6400, 7000, 500, 20, 2], 0.01, [elu, relu, elu, sigmoid],
                  [delu, drelu, delu, dsigmoid])

BATCH = 10  # Nombre de batch
EPOCH = 30  # Nombre di'mage avant retropopagation
BATCH_SIZE = 10 # Nombre d'epoch (et donc entre chaque calcul de loss)
losses = np.zeros(BATCH)
losses2 = np.zeros(BATCH)

for i in range(BATCH):
    t1 = time.time()
    for j in range(BATCH_SIZE):
        img, label = [], []
        for t in range(EPOCH):
            rn = np.random.randint(len(img_train) - 1)
            img.append(img_train[rn])
            label.append(label_train[rn])
        n.train(img, label)
    t2 = time.time()
    u = n.losses(img_test, label_test)
    losses[i] = u
    t3 = time.time()
    print(f"Batch : {i + 1}/{BATCH} en {round(t2 - t1, 4)} s avec {round(t3 - t2, 4)} en loss ({EPOCH} {BATCH_SIZE})")


plt.figure()
plt.plot(losses)
plt.xlabel("BATCH")
plt.ylabel("Loss")
plt.title(f"Covid Loss with {BATCH} batchs of {BATCH_SIZE} retropopagation")
plt.show()
