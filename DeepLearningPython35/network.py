# %load network.py
"""
network.py
~~~~~~~~~~
IT WORKS

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes, component_testing_on=0):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes

        if component_testing_on==1:
            '''component testing'''
            self.biases = [0.2+0*np.random.randn(y, 1) for y in sizes[1:]]
        else:
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        if component_testing_on==1:
            '''component testing'''
            self.weights = [0.3+0*np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        else:
            self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        print()

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            #print(b.shape)
            #print(w.shape)
            #temp = np.dot(w, a)+b
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, 
            component_testing_on=0, test_data=None ):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        '''
        for k in range(0, n, mini_batch_size):
            print(k)
            subTrainingData = training_data[k:k+mini_batch_size]
        '''

        for j in range(epochs):
            random.shuffle(training_data)

            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            
            if component_testing_on==1:
                print("==================")
                print(len(mini_batches))
                print(len(mini_batches[0]))
                print(len(mini_batches[0][0]))
                print(mini_batches[0][0])
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, component_testing_on)
                if component_testing_on==1:
                    break

            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))
            print("Got this far.")

    def update_mini_batch(self, mini_batch, eta, component_testing_on=0):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        if component_testing_on==1:
            print(len(nabla_b))
            print(len(nabla_b[0]))
            print(len(nabla_b[1]))
            print(nabla_b[0])
            print(len(nabla_b[1]))
            print(len(nabla_w))
            print(len(nabla_w[0]))
            print(len(nabla_w[1]))
            print(nabla_w[0])
            print(len(nabla_w[1]))

        iCnt=0
        for x, y in mini_batch:
            if component_testing_on==1:
                print("=====x,y=======")
                print(x,y)

            delta_nabla_b, delta_nabla_w = self.backprop(x, y, component_testing_on)
            if component_testing_on==1:
                print("===============")
                print(len(delta_nabla_b))
                print(len(delta_nabla_w))

            tempb = delta_nabla_b
            tempw = delta_nabla_w
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            iCnt = iCnt + 1

        #print("unpacked this far!!!")
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        #print()

    def backprop(self, x, y,component_testing_on=0):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        if(component_testing_on==1):
            print("======================")
            print(len(nabla_b))
            print(len(nabla_w))

        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):

            if(component_testing_on==1):
                print("======================")
                print(len(b))
                print(len(w))
                print(len(activation))

            z = np.dot(w, activation)+b
            if(component_testing_on==1):
                print("======================")
                print(len(z))

            zs.append(z)
            activation = sigmoid(z)
            if(component_testing_on==1):
                print("======================")
                print(len(activation))

            activations.append(activation)

        # backward pass
        if(component_testing_on==1):
            print("======================")
            print(len(zs[-1]))
            print(len(activations[-1]))

        tempRight = sigmoid_prime(zs[-1])
        tempLeft  = self.cost_derivative(activations[-1], y)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        if(component_testing_on==1):
            print("======================")
            print(len(delta))

        nabla_b[-1] = delta
        temp = np.dot(delta, activations[-2].transpose())
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        if(component_testing_on==1):
            print("======================")
            print(len(nabla_w[-1]))

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            temp = np.dot(self.weights[-l+1].transpose(), delta)
            if(component_testing_on==1):
                print("======================")
                print(len(sp))
                print(len(z))
                print(len(self.weights[-l+1]))
                print(len(temp))

            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            temp = np.dot(delta, activations[-l-1].transpose())
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            if(component_testing_on==1):
                print("======================")
                print(len(delta))
                print(len(sp))
                print(len(temp))

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        #c = np.array([0,1,2,3,7,5,6,5,3,2])
        #hello = np.argmax(c)
        #print(hello)
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        #print (test_results)
        c = sum(int(x == y) for (x, y) in test_results)
        #print("Percentage is - ")
        #print(c)
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
