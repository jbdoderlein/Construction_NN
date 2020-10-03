import numpy as np
from matplotlib import pyplot as plt
from NeuralNetwork import NeuralNetwork

sigmoid = lambda x: 1/(1+np.exp(-x))
dsigmoid = lambda x: np.exp(-x)/np.power(np.exp(-x)+1, 2)

n = NeuralNetwork([2, 4, 1], 0.01, [sigmoid, sigmoid], [dsigmoid, dsigmoid])
DATA = {
    "(0,0)" : (np.array([[0], [0]]), np.array([[0]])),
    "(1,0)" : (np.array([[1], [0]]), np.array([[1]])),
    "(0,1)" : (np.array([[0], [1]]), np.array([[1]])),
    "(1,1)" : (np.array([[1], [1]]), np.array([[0]])),
}
def afficher_test():
    for k, i in DATA.items():
        r = n.run(i[0])
        print(k, " -> ", r[0, 0])

def train_test():
    for k, i in DATA.items():
        n.train(*i)

print("== Resultat avant entrainement ==")
afficher_test()
for i in range(1):
    train_test()
print("== Resultat apres entrainement ==")
afficher_test()

