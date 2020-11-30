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

def construct_VGG16_Neural_Network():
    vgg16_model = tf.keras.applications.vgg16.VGG16()

    vgg16_model.summary()

    type(vgg16_model)

    ### tf.python.keras.engine.training.Model

    model_new = Sequential()
    for layer in vgg16_model.layers[:-1]:
        model_new.add(layer)

    for layer in model_new.layers:
        layer.trainable = False

    model_new.add(Dense(units=10, activation='softmax'))

    model_new.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0         
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544 
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312  
_________________________________________________________________
dense (Dense)                (None, 10)                 8194      
=================================================================
'''
def main():
    print("Hello World")
    ### The next line is for understanding the Convolutional Neural Netowrk
    ### VGG16 model proposed by K. Simonyan and A. Zisserman
    ### construct_VGG16_Neural_Network()

    model_vgg16 = Sequential()
    ### block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
    model_vgg16.add(Conv2D(name="block1_conv1", filters=64, kernel_size=(3,3), padding="same", activation="relu", input_shape=(224,224,3)))
    ### block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
    model_vgg16.add(Conv2D(name="block1_conv2", filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    ### block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
    model_vgg16.add(MaxPool2D(name="block1_pool", pool_size=(2,2),strides=(2,2)))  
    ### block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
    model_vgg16.add(Conv2D(name="block2_conv1", filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    ### block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
    model_vgg16.add(Conv2D(name="block2_conv2", filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    ### block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
    model_vgg16.add(MaxPool2D(name="block2_pool", pool_size=(2,2),strides=(2,2)))
    ### block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
    model_vgg16.add(Conv2D(name="block3_conv1", filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    ### block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
    model_vgg16.add(Conv2D(name="block3_conv2", filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    ### block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
    model_vgg16.add(Conv2D(name="block3_conv3", filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    ### block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
    model_vgg16.add(MaxPool2D(name="block3_pool", pool_size=(2,2),strides=(2,2)))
    ### block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
    model_vgg16.add(Conv2D(name="block4_conv1", filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    ### block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
    model_vgg16.add(Conv2D(name="block4_conv2", filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    ### block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
    model_vgg16.add(Conv2D(name="block4_conv3", filters=512, kernel_size=(3,3), padding="same", activation="relu"))    
    ### block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
    model_vgg16.add(MaxPool2D(name="block4_pool", pool_size=(2,2),strides=(2,2)))
    ### block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
    model_vgg16.add(Conv2D(name="block5_conv1", filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    ### block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
    model_vgg16.add(Conv2D(name="block5_conv2", filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    ### block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
    model_vgg16.add(Conv2D(name="block5_conv3", filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    ### block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
    model_vgg16.add(MaxPool2D(name="block5_pool", pool_size=(2,2),strides=(2,2)))
    ### flatten (Flatten)            (None, 25088)             0         
    model_vgg16.add(Flatten())
    ### fc1 (Dense)                  (None, 4096)              102764544 
    model_vgg16.add(Dense(name="fc1", units=4096,activation="relu"))
    ### fc2 (Dense)                  (None, 4096)              16781312  
    model_vgg16.add(Dense(name="fc2", units=4096,activation="relu"))
    ### dense (Dense)                (None, 2)                 8194      
    model_vgg16.add(Dense(name="dense", units=10, activation='softmax'))

    model_vgg16.summary()

    model_vgg16.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    train_batches, valid_batches, test_batches = hf_handler.process_data()
    model_vgg16.fit(x=train_batches,
            steps_per_epoch=len(train_batches),
            validation_data=valid_batches,
            validation_steps=len(valid_batches),
            epochs=2,
            verbose=2)

    import os.path
    if os.path.isfile('digit_vgg16_models_03_01.h5') is False:
        model_vgg16.save('models/digit_vgg16_models_03_01.h5')

    test_imgs, test_labels = next(test_batches)
    hf_handler.plotImages(test_imgs)
    print(test_labels)

    test_batches.classes
    test_batches.class_indices

    predictions = model_vgg16.predict(x=test_batches, 
        steps=len(test_batches), 
        verbose=0)

    cm = confusion_matrix(y_true=test_batches.classes, 
        y_pred=np.argmax(predictions, 
        axis=-1))

    cm_plot_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    hf_handler.plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

    print("Got this far")

'''
    The main_vgg16 impliments the AlexNet with less taining effort.
'''
def main_vgg16():
    print("Hello World")
    vgg16_model = tf.keras.applications.vgg16.VGG16()

    vgg16_model.summary()

    type(vgg16_model)

    model_new = Sequential()
    for layer in vgg16_model.layers[:-1]:
        model_new.add(layer)

    for layer in model_new.layers:
        layer.trainable = False

    model_new.add(Dense(units=10, activation='softmax'))

    model_new.summary()

    model_new.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    train_batches, valid_batches, test_batches = hf_handler.process_data()
    model_new.fit(x=train_batches,
            steps_per_epoch=len(train_batches),
            validation_data=valid_batches,
            validation_steps=len(valid_batches),
            epochs=3,
            verbose=2)

    import os.path
    if os.path.isfile('digit_vgg16_models_03.h5') is False:
        model_new.save('models/digit_vgg16_models_03.h5')

    test_imgs, test_labels = next(test_batches)
    hf_handler.plotImages(test_imgs)
    print(test_labels)

    predictions = model_new.predict(x=test_batches, steps=len(test_batches), verbose=0)

    cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
    cm_plot_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    hf_handler.plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')


if __name__ == "__main__":
    main()
    ## main_vgg16()
    print("End.")
