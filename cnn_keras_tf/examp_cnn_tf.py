import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

import train_data as my_train_data

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

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


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
if (0 != len(physical_devices)): tf.config.experimental.set_memory_growth(physical_devices[0], True)

    ### Sequantial: Group a linear stack of layers into a tensorflow keras model tepology.
    ### Dense: A regular densely connected N X N layer.
    ### Hidden layers: This layer has 16 Neurons with activation function of Relu and input layer has
    ###              the shape of (1,) or one dimensional array.
    ### Hidden layers: This layer has 32 Neurons with activation function of Relu.
    ### Output layers: This has 10 Neurons representing the number of outcomes.  the activation function
    ###              softmax provides a probability distribution among the outcomes.
    ### Dense means fully connected layer - 
model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')])

model.summary()

### Optimizer: Adam implies particular Stochastic Gradient Descent with learning rate of 0.0001
### loss: This value is specified as sparese categrical cross entropy.
model.compile(optimizer=Adam(learning_rate=0.0001), 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy'])

model.fit(x=my_train_data.scaled_train_samples, 
    y=my_train_data.train_labels, 
    validation_split=0.1, 
    batch_size=10, 
    epochs=30, 
    verbose=2)

import test_data as my_test_data

predictions = model.predict(x=my_test_data.scaled_test_samples, batch_size=10, verbose=0)  

### for i in predictions:
###     print(i)

rounded_predictions = np.argmax(predictions, axis=-1)

### for i in rounded_predictions:
###     print(i)

### The confusion matrix we’ll be plotting comes from scikit-learn.
### We then create the confusion matrix and assign it to the variable cm. T
cm = confusion_matrix(y_true=my_test_data.test_labels, y_pred=rounded_predictions)

cm_plot_labels = ['no_side_effects','had_side_effects']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

import os.path
if os.path.isfile('medical_trial_models.h5') is False:
    model.save('models/medical_trial_model.h5')

from tensorflow.keras.models import load_model
new_model = load_model('models/medical_trial_model.h5')

new_model.summary()
new_model.get_weights()

print("Got this far")