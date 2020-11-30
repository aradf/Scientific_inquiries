import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import helper_func as hf_handler

def initialize():
    print("initialize!")
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    if (0 != len(physical_devices)): tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main():
    print("Hello World!")
    initialize()
    train_batches, valid_batches, test_batches = hf_handler.process_data()

    ### Sequantial: Group a linear stack of layers into a tensorflow keras model tepology.
    ### Conv2D: A 2-Dim Convolutional Layer. It will have 32 output filters each with a kernel
    ###         size of 3x3, and Relu Activation function. Padding of 'same' enables zero-padding.
    ###         The input_shape data which is specified on the first layer only.  In this case,
    ###         The image is 224 pixels high and 224 pixels wide with 3 color chanels (RGB)
    ### MaxPool2D: The 2-Dim max pooling layer reduces the dimension of the data.
    ### Conv2D: This layer has a filter of size 64 and no input_shape
    ### MaxPool2D: The 2-Dim max pooling layer reduces the dimension of the data.
    ### Flatten: Flatten the convolutiona layer and pass it to dense layer.
    ### Dense: This is the output layer of the network, and has 10 nodes which is the number of 
    ### of output or classification. It uses the softmax activation function on the output, such that
    ### each sample is a probability distribution over the outputs.
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(224,224,3)),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Flatten(),
        Dense(units=10, activation='softmax')
    ])

    model.summary()

    ### Optimizer: Adam implies particular Stochastic Gradient Descent with learning rate of 0.0001
    ### loss: This value is specified as sparese categrical cross entropy.
    model.compile(optimizer=Adam(learning_rate=0.0001), 
        loss='categorical_crossentropy', 
        metrics=['accuracy'])

    model.fit(x=train_batches, 
        steps_per_epoch=len(train_batches),
        validation_data=valid_batches,
        epochs=3,
        verbose=2
    )

    import os.path
    if os.path.isfile('digit_trial_models_01.h5') is False:
        model.save('models/digit_trial_models_01.h5')

    test_imgs, test_labels = next(test_batches)
    hf_handler.plotImages(test_imgs)
    print(test_labels)

    test_batches.classes
    test_batches.class_indices

    predictions = model.predict(x=test_batches, 
        steps=len(test_batches), 
        verbose=0)

    cm = confusion_matrix(y_true=test_batches.classes, 
        y_pred=np.argmax(predictions, 
        axis=-1))

    cm_plot_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    hf_handler.plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')


if __name__ == "__main__":
    main()
    print("Got this far.")
