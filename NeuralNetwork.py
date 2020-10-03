import numpy as np

class NeuralNetwork:
    """"Réseaux de neurones"""
    def __init__(self, schema, learning_rate, activation, derivation_activation):
        self.schema = schema
        self.learning_rate = learning_rate
        self.activation = activation
        self.derivation_activation = derivation_activation
        self.generate_weights()

    def generate_weights(self):
        self.weights = []
        for i in range(len(self.schema)-1):
            matrice = np.random.random(size=(self.schema[i+1], self.schema[i]+1))
            self.weights.append(matrice)

    def run(self, ndinput):
        result = ndinput
        for i in range(len(self.weights)):
            result = self.activation[i](np.dot(self.weights[i], result))
        return result

    def train(self, ndinput, ndoutput):
        n = len(self.schema)
        X = [] # neurone activé
        H = [] # neurone avant activation
        E = [] # Perte
        # == Propagation avant avec stockage des résultats intermediaire pour calcul de perte ==
        #Premier couche avec biais pour initialiser
        result = ndinput
        X.append(result)
        for i in range(len(self.weights)):
            h = np.dot(self.weights[i], result)
            result = self.activation[i](h)
            X.append(result)
            H.append(h)
        # == Calcul des pertes ==
        e = self.derivation_activation[-1](H[-1]) * (X[-1] - ndoutput) # * car c est pour chaque ligne i
        E.append(e)

        for i in range(n-2): # On aura n-1 matrice E, on a deja fait celle n-1, il nosu en reste n-2
            ni = n-2+i # coeff inverse car on va a l envers (on utilisera i pour E car on le retournera a la fin
            e = self.derivation_activation[ni-1](H[ni-1]) * (np.dot(np.transpose(self.weights[ni]), E[i]))
            E.append(e)
        E.reverse()
        # == Modification des poids ==

        for l in range(n-1):
            self.weights[l] = self.weights[l] - self.learning_rate * np.dot(E[l], np.transpose(X[l]))



