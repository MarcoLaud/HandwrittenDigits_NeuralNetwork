"""
Implementation of a three-layers Neural Network (NN) aimed at recognizing
handwritten digits from 0 to 9.

Training and test database: MNIST (http://yann.lecun.com/exdb/mnist/)

@author: Marco Laudato
"""

from idx2numpy import convert_from_file
from matplotlib.pyplot import imshow, subplot, subplots
from numpy import concatenate, c_, dot, exp, load, log, matmul, ones, r_,\
                  random, save, sqrt, tensordot, transpose, zeros
from random import sample
from scipy.optimize import fmin_l_bfgs_b


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
                           self.epsilon_random_gen * 2) -\
            self.epsilon_random_gen

        # weigths matrix from second to third layer
        self.weigths_23 = (random.rand(self.number_of_features,
                           self.number_of_hidden_units + 1) *
                           self.epsilon_random_gen * 2) -\
            self.epsilon_random_gen

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

    def dataDisplay(self, n=16):
        """
        Picks n elements from the MNIST training dataset and display the
        corresponding images. n is a perfect square.
        """
        rand_pics = sample(self.extension_of_dataset, n)
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

    def forwardPropagation(self, D, m, W_1, W_2):
        """
        Implements forward propagation of the element m of the dataset D.
        W_1 and W_2 are weights matrices (reshape if in vector form!)
        """
        # first to second layer:
        z_12 = matmul(W_1, transpose(D[m]))
        a_12 = self.sigmoid(z_12)
        # append 1 to the second layer activation nodes:
        a_12 = r_[1, a_12]

        # second to third layer:
        z_23 = matmul(W_2, transpose(a_12))
        self.h = self.sigmoid(z_23)            # final result
        return [self.h, a_12]

    def costFunction(self, W):
        """
        Computes the cost function given the weights.
        W is a weigths vector. It is the concatenation of the two unrolled
        vectors [w1, w2] relative to the weigths matrices.
        """
        J = 0   # accumulator for cost function
        num_of_rows_w12 = self.weigths_12.shape[0]
        num_of_cols_w12 = self.weigths_12.shape[1]
        num_of_rows_w23 = self.weigths_23.shape[0]
        num_of_cols_w23 = self.weigths_23.shape[1]

        len_w1 = num_of_rows_w12 * num_of_cols_w12

        # reshaping the weights matrices:
        W1 = W[:len_w1].reshape(num_of_rows_w12, num_of_cols_w12)
        W2 = W[len_w1:].reshape(num_of_rows_w23, num_of_cols_w23)

        for mm in range(self.extension_of_dataset):
            Y = self.y[mm]
            H = self.forwardPropagation(self.X, mm, W1, W2)[0]
            J += -Y.dot(log(H)) - (1 - Y).dot(log(1 - H))
        return (J / self.extension_of_dataset)

    def backPropagation(self, W):
        """
        Computes back propagation and return gradient vector for optimization
        """
        # accumulator for gradient matrix relative to weigts_12
        grad_1 = zeros((self.number_of_hidden_units,
                        self.number_of_pixels ** 2 + 1))

        # accumulator for gradient matrix relative to weights_23
        grad_2 = zeros((self.number_of_features,
                        self.number_of_hidden_units + 1))

        # reshaping the weights matrices:
        num_of_rows_w12 = self.weigths_12.shape[0]
        num_of_cols_w12 = self.weigths_12.shape[1]
        num_of_rows_w23 = self.weigths_23.shape[0]
        num_of_cols_w23 = self.weigths_23.shape[1]

        len_w1 = num_of_rows_w12 * num_of_cols_w12
        len_w2 = num_of_rows_w23 * num_of_cols_w23

        W1 = W[:len_w1].reshape(num_of_rows_w12, num_of_cols_w12)
        W2 = W[len_w1:].reshape(num_of_rows_w23, num_of_cols_w23)

        for mm in range(self.extension_of_dataset):
            H, a_12 = self.forwardPropagation(self.X, mm, W1, W2)
            delta_3 = H - self.y[mm]
            delta_2 = matmul(W2.T, delta_3) * a_12 * (1 - a_12)
            # removing the first element:
            delta_2 = delta_2[1:]
            # computing the gradient matrices:
            grad_1 += tensordot(delta_2, self.X[mm], axes=0)
            # axes = 0 -> tensor product
            grad_2 += tensordot(delta_3, a_12.T, axes=0)
            # transposed to match dimensions

        # normalization:
        grad_1 = grad_1 / self.extension_of_dataset
        grad_2 = grad_2 / self.extension_of_dataset

        # unrolling the gradients as a vector for optimization:
        g1 = grad_1.reshape(len_w1)
        g2 = grad_2.reshape(len_w2)
        grad = concatenate((g1, g2))

        return grad

    def gradChecking(self, W, grad, n=5, epsilon=10**(-4)):
        """
        Compute the mean relative difference between the gradient of the cost
        function J computed by back propagation and via finite differences in n
        different points.
        """
        rnd_indeces = sample(range(len(grad)), n)
        finite_diffs = zeros(n)
        rnd_gradients = [grad[ii] for ii in rnd_indeces]
        error = 0

        for ii, index in enumerate(rnd_indeces):
            W[index] += epsilon
            J_plus = self.costFunction(W)
            W[index] -= 2*epsilon
            J_minus = self.costFunction(W)
            W[index] += epsilon
            finite_diffs[ii] = (J_plus - J_minus) / (2 * epsilon)

        for ii in range(n):
            if rnd_gradients[ii] == 0.0 or finite_diffs[ii] == 0.0:
                n -= 1
                pass
            else:
                error += abs((rnd_gradients[ii] - finite_diffs[ii]) /
                             (rnd_gradients[ii] + finite_diffs[ii]))

        return error / n

    def training(self):
        """
        Computes the optimal values of the weigths.
        """
        # some useful quantities
        num_of_rows_w12 = self.weigths_12.shape[0]
        num_of_cols_w12 = self.weigths_12.shape[1]
        num_of_rows_w23 = self.weigths_23.shape[0]
        num_of_cols_w23 = self.weigths_23.shape[1]

        len_w1 = num_of_rows_w12 * num_of_cols_w12
        len_w2 = num_of_rows_w23 * num_of_cols_w23

        # unrolling weight matrices
        w1 = self.weigths_12.reshape(len_w1)
        w2 = self.weigths_23.reshape(len_w2)
        W = concatenate((w1, w2))

        # find optimal weigths to minimize the cost function
        W_opt = fmin_l_bfgs_b(self.costFunction, W,
                              fprime=self.backPropagation, maxiter=300)

        # writing and saving optimal weigth matrices
        w1_opt = W_opt[0][:len_w1].reshape(num_of_rows_w12, num_of_cols_w12)
        w2_opt = W_opt[0][len_w1:].reshape(num_of_rows_w23, num_of_cols_w23)

        save('w1_opt', w1_opt)
        save('w2_opt', w2_opt)

        return w1_opt, w2_opt

    def testAccuracy(self, path_to_testData, path_to_testLabels,
                     tolerance=0.5):
        """
        Computes the accuracy of the NN on the test datasets.
        """
        # loading the optimized weigth matrices
        w1 = load('w1_opt.npy')
        w2 = load('w2_opt.npy')

        # loading the training set and training lables
        test_set = convert_from_file(path_to_testData)
        test_set_labels = convert_from_file(path_to_testLabels)

        # defining test matrix and results (Z) matrix
        T = zeros((test_set.shape[0], test_set.shape[1] ** 2))
        Z = zeros(test_set.shape[0])

        # populating the test matrix
        for ii in range(test_set.shape[0]):
            T[ii] = test_set[ii].reshape(test_set.shape[1] ** 2)

        # appending the bias x_0=1
        T = c_[ones(test_set.shape[0]), T]

        res = 0
        counter = 0

        # filtering results of test set to [0,1] vectors
        for mm in range(test_set.shape[0]):
            result = self.forwardPropagation(T, mm, w1, w2)[0]
            for ii, num in enumerate(result):
                if num > tolerance:
                    result[ii] = 1
                else:
                    result[ii] = 0
        # transforming result in a single digit
            for ii, num in enumerate(result):
                res += ii * num
            Z[mm] = res
            res = 0

        # computing the accuracy
        for mm in range(test_set.shape[0]):
            if Z[mm] - test_set_labels[mm] != 0:
                counter += 1
        accuracy = (test_set.shape[0] - counter) / test_set.shape[0]
        return accuracy * 100
