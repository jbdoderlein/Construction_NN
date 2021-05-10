import numpy as np
import dill
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image


class ConvolutionLayer:
    def __init__(self, fs, es):
        """
        A convolution layer
        :param fs: size of layer
        :param es: size of entry image
        """
        self.weight = np.random.randn(*fs) * 0.1
        self.bias = np.random.randn(1)[0] * 0.1
        self.es = es

    def forward(self, image: np.array) -> np.array:
        l, h = image.shape  # dimension de l'image
        lz = filter.shape[0]  # largueur du filtre
        sz = (lz - 1)  # taille du filtre
        result = np.zeros(image.shape)
        for i in range(l - sz):
            for j in range(h - sz):
                result[i, j] = np.sum(image[i:i + lz, j:j + lz] * self.weight)
        return result + self.bias

    @property
    def size(self):
        """
        Give entry and output size of the layer
        :return:
        """
        return self.es, self.es-(self.fs[0]//2 + 1)




class ConvolutionalNeuralNetwork:
    """"Réseaux de neurones convolutionel"""

    def __init__(self, layers_config, input_size, output_size):
        self.layers_config = layers_config
        self.input_size = input_size
        self.output_size = output_size
        self.layer = []
        self.init_layers()

    def init_layers(self):
        fname, farg = self.layers_config[0]
        self.layer.append(self.gen_layer(fname, self.input_size, farg))
        for i in range(1,len(self.layers_config)):
            fname, farg = self.layers_config[i]
            self.layer.append(self.gen_layer(fname, self.layer[-1].size()[1], farg))
        self.layer.append(self.gen_layer('dense', self.layer[-1].size()[1], farg))


    def gen_layer(self, name, inp, arg):
        if name == "convolution":
            pass
        if name == "max_pooling":
            pass
        if name == "dropout":
            pass
        if name == "flatten":
            pass
        if name == "activation":
            pass


    def forward(self, image: np.array):
        l, h = image.shape  # dimension de l'image
        lz = filter.shape[0]  # largueur du filtre
        sz = (lz - 1)  # taille du filtre
        result = np.zeros((l - sz, h - sz))
        for i in range(l - sz):
            for j in range(h - sz):
                result[i, j] = np.sum(image[i:i + lz, j:j + lz] * filter)
        return result


if __name__ == '__main__':
    cnn = ConvolutionalNeuralNetwork([])
    img = np.random.random((3,3))
    nimg = cnn.apply_filter(img, f1)
    plt.imshow(nimg, cmap=cm.gray, interpolation=None)
    plt.show()
