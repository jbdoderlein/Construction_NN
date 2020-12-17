import cupy as np
import dill
import time

class NeuralNetwork:
    """"Réseaux de neurones"""
    def __init__(self, schema=[], learning_rate=0.1, activation=[], derivation_activation=[], load=None):
        if not load:
            self.schema = schema
            self.activation = activation
            self.learning_rate = learning_rate
            self.derivation_activation = derivation_activation
            self.generate_weights()
            self.generate_biases()
        else:
            self.load(load)

    def save(self, filename):
        data = {
            'schema': self.schema,
            'activation': self.activation,
            'derivation_activation': self.derivation_activation,
            'learning_rate': self.learning_rate,
            'weights': self.weights,
            'biases': self.biases
        }
        with open(filename, 'wb') as f:
            dill.dump(data,f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            nn = dill.load(f)
        self.schema = nn["schema"]
        self.activation = nn["activation"]
        self.learning_rate = nn["learning_rate"]
        self.derivation_activation = nn["derivation_activation"]
        self.weights = nn["weights"]
        self.biases = nn["biases"]

    def generate_weights(self):
        self.weights = []
        for i in range(len(self.schema)-1):
            matrice = np.random.normal(0, 1, size=(self.schema[i+1], self.schema[i]))
            self.weights.append(matrice)

    def generate_biases(self):
        self.biases = []
        for i in range(len(self.schema)-1):
            matrice = np.zeros(shape=(self.schema[i+1], 1))
            self.biases.append(matrice)

    def forward_propagation(self, vector_input):
        result = vector_input.reshape(vector_input.shape[0], 1) # on met en colonne
        for i in range(len(self.weights)):
            result = self.activation[i](np.dot(self.weights[i], result) + self.biases[i])
        return result

    def train(self, train_data, train_labels):
        s = len(self.weights)  # Nombre de matrice de poids (et aussi de biais)
        m = len(train_data)  # Taille du set d'entrainement
        dW = [0]*s
        dB = [0]*s
        for i in range(m):
            X = []
            Z = []
            #Propagation avant
            result = train_data[i].reshape(train_data[i].shape[0], 1)
            X.append(result)
            for j in range(s):
                z = np.dot(self.weights[j], result) + self.biases[j]
                result = self.activation[j](z)
                X.append(result)
                Z.append(z)
            #Retro propagation
            dz = X[-1] - train_labels[i]
            dW[-1] += np.dot(dz, X[-2].T)
            dB[-1] += dz
            for j in range(2, s+1):
                da = np.dot(self.weights[-j+1].T, dz)
                dz = np.multiply(da, self.derivation_activation[-j](X[-j]))
                dW[-j] = dW[-j] + np.dot(dz, X[-j-1].T)
                dB[-j] = dB[-j] + dz
        #Application aux biais et aux poids
        for j in range(s):
            self.biases[j] -= self.learning_rate * (dB[j] / m)
            self.weights[j] -= self.learning_rate * (dW[j] / m)

    def losses(self, training_data, training_labels):
        """
        Erreur quadratique moyenne
        :param training_data: liste de tests
        :param training_labels: resultats attendu des test
        :return: erreur quadratique moyenne
        """
        loss = 0
        n = len(training_data)
        for i in range(n):
            result = self.forward_propagation(training_data[i])
            loss += np.mean((result - training_labels[i]) ** 2)
        return loss/n



