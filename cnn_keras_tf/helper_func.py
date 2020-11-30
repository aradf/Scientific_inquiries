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

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        img = normalize(img)
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()

    plt.show()

def process_data():
    print("collection_data!")
    train_path = 'data/digits/train'
    valid_path = 'data/digits/valid'
    test_path = 'data/digits/test'

    ### ImageDataGenerator() creates train, test, and validation batchs from the directories.
    ### 

    train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory=train_path, 
        target_size=(224,224), 
        classes=['0', '1', '2', '3', '4', '5', '6', '7', '8','9'], 
        batch_size=10)

    valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory=valid_path, 
        target_size=(224,224), 
        classes=['0', '1', '2', '3', '4', '5', '6', '7', '8','9'], 
        batch_size=10)

    test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory=test_path, 
        target_size=(224,224), 
        classes=['0', '1', '2', '3', '4', '5', '6', '7', '8','9'], 
        batch_size=10, 
        shuffle=False)

    assert train_batches.n == 42000
    assert valid_batches.n == 600
    assert test_batches.n == 350
    assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 10

    imgs, labels = next(train_batches)

    plotImages(imgs)
    print(labels)
    return(train_batches, valid_batches, test_batches)
