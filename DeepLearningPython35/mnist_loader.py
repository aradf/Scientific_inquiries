# %load mnist_loader.py
"""
mnist_loader
~~~~~~~~~~~~
A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])

    '''
    Utilzied for C++ converstion.


    tmpfile1 = "/home/faramarz/Desktop/ArtNeuralNet/dataset/training_data.bin"
    tmpfile2 = "/home/faramarz/Desktop/ArtNeuralNet/dataset/validation_data.bin"
    tmpfile3 = "/home/faramarz/Desktop/ArtNeuralNet/dataset/test_data.bin"

    import sys
    import struct
    handle1 = open(tmpfile1 , 'wb')
    for l1, n1 in training_data:
        handle1.write(struct.pack('<%df' % len(l1), *l1))
        handle1.write(struct.pack('<%df' % len(n1), *n1))

    handle1.close()

    handle2 = open(tmpfile2 , 'wb')
    for l2, n2 in validation_data:
        handle2.write(struct.pack('<%df' % len(l2), *l2))
        #handle2.write(struct.pack('<%df' % len(n2), *n2))
        handle2.write(struct.pack('<%df' % n2.size, *[n2]))

    handle2.close()

    handle3 = open(tmpfile3 , 'wb')
    test_data_dump = list(test_data)
    for l3, n3 in test_data_dump:
        handle3.write(struct.pack('<%df' % len(l3), *l3))
        #handle3.write(struct.pack('<%df' % len(n3), *n3))
        import numpy
        a3 = numpy.zeros(10,dtype=float)
        a3[n3] = 1
        handle3.write(struct.pack('<%df' % len(a3), *a3))

    handle3.close()
    '''
        
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
