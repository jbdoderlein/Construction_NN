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
            matrice = np.random.random(size=(self.schema[i]+1, self.schema[i+1]))
            self.weights.append(matrice)

    def run(self, ndinput):
        result = np.column_stack((ndinput, np.array([[1]])))
        for i in range(len(self.weights)-1):
            result = np.column_stack(
                (self.activation[i](np.dot(result, self.weights[i])), np.array([[1]]))
            )
        return self.activation[-1](np.dot(result, self.weights[-1]))

    def train(self, ndinput, ndoutput):
        n = len(self.schema)
        X = [] # neurone activé
        H = [] # neurone avant activation
        E = [] # Perte
        # == Propagation avant avec stockage des résultats intermediaire pour calcul de perte ==
        #Premier couche avec biais pour initialiser
        result = np.column_stack((ndinput, np.array([[1]])))
        X.append(result)
        for i in range(len(self.weights) - 1): # Les n-2 couches suivante (avec le biais)
            h = np.dot(result, self.weights[i])
            result = np.column_stack((self.activation[i](h), np.array([[1]])))
            print(result)
            H.append(h)
            X.append(result)
        # Derniere couche sans biais en sortie
        h = np.dot(result, self.weights[-1])
        H.append(h)
        X.append(self.activation[-1](h))
        # == Calcul des pertes ==
        e = np.dot(np.transpose(self.derivation_activation[-1](H[-1])), (X[-1] - ndoutput)) # derniere couche
        E.append(e)
        for i in range(n-2): # On aura n-1 matrice E, on a deja fait celle n-1, il nosu en reste n-2
            ni = n-2+i # coeff inverse car on va a l envers (on utilisera i pour E car on le retournera a la fin
            e = self.derivation_activation[ni-1](H[ni-1]) * (np.dot(self.weights[ni], E[i]))
            E.append(e)
        E.reverse()
        # == Modification des poids ==
        """
        for l in range(n-1):
            self.weights[l] = self.weights[l] - self.learning_rate * np.dot(E[l], X[l])
        """
        return X, H, E


