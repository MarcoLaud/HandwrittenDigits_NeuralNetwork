"""
Implementation of a Neural Network (NN) aimed at recognizing handwritten digits
from 0 to 9.

@author: Marco Laudato
"""

"""
The NN architecture is made of three layers. Input layer, one hidden layer and
the output layer.

The dataset that we are using to train the NN is made of 28 x 28 px images.
So the first layer will have 784 units. The hidden layer has 25 units.
The output layer has 10 units since the training set admits only 10 digits from
0 to 9.

The main steps of the algorithm are:
    1) Define all the variables
    2) Randomly initialize the parameter matrices.
    3) Forward propagation to compute the initial hypothesis value.
    4) Back propagation to compute the gradient.
    5) Check the gradient value with finite difference (then disable it).
    6) Use gradient descent with BP to determine optimal parameters.
"""

from idx2numpy import convert_from_file
from matplotlib.pyplot import imshow, subplot, subplots
from numpy import c_, dot, exp, log, matmul, ones, r_, random, sqrt,\
                  transpose, zeros
from random import sample


class NeuralNetwork:

    def __init__(self, training_dataset_path, labels_dataset_path):
        self.learning_set = convert_from_file(training_dataset_path)
        self.labels = convert_from_file(labels_dataset_path)
        self.extension_of_dataset = self.learning_set.shape[0]
        self.number_of_pixels = self.learning_set.shape[1]
        self.number_of_hidden_units = 25
        self.number_of_features = 10
        self.epsilon_random_gen = 0.1

        # weigths matrix from first to second layer
        self.weigths_12 = (random.rand(self.number_of_hidden_units,
                           self.number_of_pixels ** 2 + 1) *
                         self.epsilon_random_gen * 2) - self.epsilon_random_gen

        # weigths matrix from second to third layer
        self.weigths_23 = (random.rand(self.number_of_features,
                           self.number_of_hidden_units + 1) *
                         self.epsilon_random_gen * 2) - self.epsilon_random_gen

        # learning dataset matrix
        self.X = zeros((self.extension_of_dataset, self.number_of_pixels ** 2))
        # population of dataset matrix
        for ii in range(self.extension_of_dataset):
            self.X[ii] = \
                self.learning_set[ii].reshape(self.number_of_pixels ** 2)
        # appending the bias unit vector x_0 = 1
        self.X = c_[ones(self.extension_of_dataset), self.X]

        # labels in vector form
        self.y = zeros((self.extension_of_dataset, self.number_of_features))
        # converting from single digit to vector form
        for ii, num in enumerate(self.labels):
            self.y[ii, num] = 1

        # argument of the sigmoid function from first to second layer
        self.z_12 = zeros(self.number_of_hidden_units)

        # argument of the sigmoid function from second to third layer
        self.z_23 = zeros(self.number_of_features)

        # values vector of the second layer's attivation nodes
        self.a_12 = zeros(self.number_of_hidden_units)

        # values vector of the third layer's attivation nodes (hypothesis)
        self.h = zeros(self.number_of_features)

        # initial cost value set to zero
        self.J = 0

    def dataDisplay(self, n=16):
        """
        Picks n elements from the MNIST training dataset and display the
        corresponding images. n is a perfect square.
        """
        rand_pics = sample(range(self.extension_of_dataset), n)
        edge_size = int(sqrt(n))

        fig, axes = subplots(edge_size, edge_size)
        fig.suptitle('Examples from the training dataset')

        for ii, idx in enumerate(rand_pics):
            ax = subplot(edge_size, edge_size, ii+1)
            imshow(self.learning_set[idx], cmap='gray', vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])

    def sigmoid(self, x):
        """
        Computes the values vector of the sigmoid function, when x is an array
        """
        return 1 / (1 + exp(-x))

    def forwardPropagation(self, m):
        """
        Implements forward propagation of the dataset element m
        """
        # first to second layer:
        self.z_12 = matmul(self.weigths_12, transpose(self.X[m]))
        self.a_12 = self.sigmoid(self.z_12)
        # append 1 to the second layer activation nodes:
        self.a_12 = r_[1, self.a_12]

        # second to third layer:
        self.z_23 = matmul(self.weigths_23, transpose(self.a_12))
        self.h = self.sigmoid(self.z_23)            # final result
        return self.h

    def costFunction(self):
        """
        Computes the cost function given the weights
        """
        for mm in range(self.extension_of_dataset):
            Y = self.y[mm]
            H = self.forwardPropagation(mm)
            self.J += -Y.dot(log(H)) - (1 - Y).dot(log(1 - H))
            self.J = self.J / self.extension_of_dataset
