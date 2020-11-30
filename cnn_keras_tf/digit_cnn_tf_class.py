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
from tensorflow.keras.models import load_model

def main():
    train_batches, valid_batches, test_batches = hf_handler.process_data()
    model_loaded = load_model('models/digit_trial_models_01.h5')

    predictions = model_loaded.predict(x=test_batches, 
            steps=len(test_batches), 
            verbose=2)

    print(test_batches)
    print(predictions)

    for iCnt in range(len(predictions)):
        print("----------")
        print("prediction : ", predictions[iCnt])
        print("test_batches.classes : ", test_batches.classes[iCnt])

    cm = confusion_matrix(y_true=test_batches.classes, 
                        y_pred=np.argmax(predictions, 
                        axis=-1))

    cm_plot_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    hf_handler.plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

def main_vgg16():
    train_batches, valid_batches, test_batches = hf_handler.process_data()
    
    vgg16_model = tf.keras.applications.vgg16.VGG16()
    vgg16_model.summary()
    type(vgg16_model)
    model_loaded = Sequential()
    for layer in vgg16_model.layers[:-1]:
        model_loaded.add(layer)

    for layer in model_loaded.layers:
        layer.trainable = False

    model_loaded.add(Dense(units=10, activation='softmax'))
    model_loaded.summary()
    model_loaded.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    model_loaded.load_weights('models/digit_vgg16_models_03.h5')
    predictions = model_loaded.predict(x=test_batches, 
            steps=len(test_batches), 
            verbose=2)

    print(test_batches)
    print(predictions)

    for iCnt in range(len(predictions)):
        print("----------")
        print("prediction : ", predictions[iCnt])
        print("test_batches.classes : ", test_batches.classes[iCnt])

    cm = confusion_matrix(y_true=test_batches.classes, 
                        y_pred=np.argmax(predictions, 
                        axis=-1))

    cm_plot_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    hf_handler.plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

if __name__ == "__main__":
    ### main()
    main_vgg16()

    print("Got this far.")
