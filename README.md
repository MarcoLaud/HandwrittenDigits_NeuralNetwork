This is a Numpy/Scipy implementation of the three-layers Neural Network presented in the Andrew NG's Coursera lectures.

The Neural Network (NN) architecture is made of three layers. The input layer, one hidden layer and the output layer.

I am using the MNIST database 

(http://yann.lecun.com/exdb/mnist/)

which is made of 60k (28x28 px) images.

Consequently, the first layer has 784 units, the hidden layer has 25 units. The output layer has 10 units since the training set admits only 10 digits from 0 to 9.

The main steps of the algorithm are:
    1) Define all the variables
    2) Randomly initialize the parameter matrices.
    3) Forward propagation to compute the initial hypothesis value.
    4) Back propagation to compute the gradient.
    5) Check the gradient value with finite difference.
    6) Use gradient descent with BP to determine optimal parameters.
    7) Compute the accuracy.

The optimal weigth matrices have been saved in npy format.

The accuracy relative to the current implementation is 86.42%, computed on the test set of the MNIST database.

Hope you like!
Marco
name AT namesurname.com

---
To be implemented:
1) regularized cost function.
2) visualization of what the hidden layer is learning.  