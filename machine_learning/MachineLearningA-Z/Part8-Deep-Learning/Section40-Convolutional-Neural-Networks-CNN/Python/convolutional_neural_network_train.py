# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tf.__version__

# https://keras.io/examples/

# Part 1 - Data Preprocessing
# Preprocessing the Training set
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
# When resacle is set to some value, rescaling is applied to sample data before computing internal data.
# shear_range controls the amount of shear transformation applied to images.
# zoom_range specifies the amount of zooming applied to the data.
train_dataGenerator = ImageDataGenerator(rescale = 1./255,
                                         shear_range = 0.2,
                                         zoom_range = 0.2,
                                         horizontal_flip = True)

# target size is the dimensions to which the images are resized.
# batch_size is the number of samples (images) processed before the model is updated.
# class_mode could be 'catgorical', 'binary', 'sparse' or 'None'. Binary means one-dim binary label.
training_set = train_dataGenerator.flow_from_directory('dataset/training_set',
                                                        target_size = (64, 64),
                                                        batch_size = 32,
                                                        class_mode = 'binary')

# Preprocessing the Test set
test_dataGenerator = ImageDataGenerator(rescale = 1./255)
test_set = test_dataGenerator.flow_from_directory('dataset/test_set',
                                                   target_size = (64, 64),
                                                   batch_size = 32,
                                                   class_mode = 'binary')

# Part 2 - Building the CNN
# Initialising the CNN
cnn_model = tf.keras.models.Sequential()

# Step 1 - Convolution
# The filters paramter sets the number of output filters in the convlutional layer.
# kerne_size is the spatial dimention of the kernel.
# activation sets the activation function for the convlutional layer.
# input_shape sets the dimentions of the input image.
cnn_model.add(tf.keras.layers.Conv2D( filters=32, 
                                      kernel_size=3, 
                                      activation='relu', 
                                      input_shape=[64, 64, 3]))

# Step 2 - Pooling
# pool_size determines the size of the pool windows.
# strides set the length of the stride.
cnn_model.add(tf.keras.layers.MaxPool2D( pool_size=2, 
                                         strides=2 ))

# Adding a second convolutional layer
cnn_model.add(tf.keras.layers.Conv2D( filters=32, 
                                      kernel_size=3, 
                                      activation='relu'))

cnn_model.add(tf.keras.layers.MaxPool2D( pool_size=2, 
                                         strides=2 ))

# Step 3 - Flattening
cnn_model.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn_model.add(tf.keras.layers.Dense( units=128, 
                                     activation='relu' ))

# Step 5 - Output Layer
cnn_model.add(tf.keras.layers.Dense( units=1, 
                                     activation='sigmoid' ))

# Part 3 - Training the CNN
# Compiling the CNN
# optimizion al gori thm is 'adam' means 'Adaptive Moment Estimation' 
# metrics measures the pcercentage of correct prediction a model makes.
cnn_model.compile( optimizer = 'adam', 
                   loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn_model.fit( x = training_set, 
               validation_data = test_set, 
               epochs = 17)

# cnn_model.fit( x = training_set, 
#             validation_data = test_set, 
#             epochs = 25)
cnn_model.save('cnn_keras.keras')

# Conver models to onnx and export.

import tf2onnx
import onnxruntime
import onnx

spec = (tf.TensorSpec((None, 64, 64, 3), 
                       tf.float32, 
                       name="input"), )

output_path = "cnn_onnx.keras.onnx"
model_onnx, _ = tf2onnx.convert.from_keras( model = cnn_model, 
                                            input_signature = spec,
                                            opset = 13,
                                            output_path = output_path)

# python3 -m tf2onnx.convert --sved-model cnn_onnx.keras --output cnn_onnx.keras.onnx -opset 13
onnx.save_model(model_onnx, output_path)

# Part 4 - Making a single prediction
import numpy as np
from tensorflow.keras.preprocessing import image
# test_image = image.load_img( 'dataset/single_prediction/cat_or_dog_1.jpg', 
#                              target_size = (64, 64) )

test_image = image.load_img( 'dataset/single_prediction/cat_or_dog_2.jpg', 
                             target_size = (64, 64) )

test_image = image.img_to_array(test_image)
test_image = np.expand_dims( test_image, 
                             axis = 0)
result = cnn_model.predict( test_image )
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)

# save the models
# https://keras.io/guides/serialization_and_saving/
# https://onnxruntime.ai/docs/get-started/with-python.html
reconstructed_cnnModel = tf.keras.models.load_model('cnn_keras.keras')
new_result = reconstructed_cnnModel.predict( test_image )
if new_result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)

print("hello")