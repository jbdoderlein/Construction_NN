import mnist
from sklearn import svm
import numpy as np

clf = svm.SVC(gamma=0.001)

x_train, t_train, x_test, t_test = mnist.load()

N_TRAIN = 500
N_TEST = 20
training_data = [x_train[i].reshape((28, 28)) for i in range(N_TRAIN)]
training_labels = t_train[:N_TRAIN]

clf.fit(training_data, training_labels)

print("test :", t_test[0])
print("predicted", clf.predict(x_test[0].reshape((28, 28))))
