# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
print( tf.__version__ )

# https://keras.io/examples/
# https://www.linkedin.com/pulse/everything-you-need-know-vgg16-deep-learning-syed-talal-musharraf-pjkkf/

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
                                                        target_size = (224, 224),
                                                        batch_size = 32,
                                                        class_mode = 'categorical')

# Preprocessing the Test set
test_dataGenerator = ImageDataGenerator(rescale = 1./255)
test_set = test_dataGenerator.flow_from_directory('dataset/test_set',
                                                   target_size = (224, 224),
                                                   batch_size = 32,
                                                   class_mode = 'categorical')

# Part 2 - Building the vgg16
# Initialising the vgg16
vgg16_model = tf.keras.models.Sequential()

# Step 1 - Convolution
# The filters paramter sets the number of output filters in the convlutional layer.
# kernel_size is the spatial dimention of the kernel.
# activation sets the activation function for the convlutional layer.
# input_shape sets the dimentions of the input image.
vgg16_model.add(tf.keras.layers.Conv2D( filters=64, 
                                        kernel_size=3,
                                        padding='same', 
                                        activation='relu', 
                                        input_shape=[224, 224, 3]))

vgg16_model.add(tf.keras.layers.Conv2D( filters=64,
                                        kernel_size=3,
                                        padding='same', 
                                        activation='relu'))

# Step 2 - Pooling
# pool_size determines the size of the pool windows.
# strides set the length of the stride.
vgg16_model.add(tf.keras.layers.MaxPool2D( pool_size=2, 
                                           strides=2 ))


vgg16_model.add(tf.keras.layers.Conv2D( filters=128, 
                                        kernel_size=3, 
                                        padding='same', 
                                        activation='relu') )

vgg16_model.add(tf.keras.layers.Conv2D( filters=128, 
                                        kernel_size=3, 
                                        padding='same', 
                                        activation='relu') )


vgg16_model.add(tf.keras.layers.MaxPool2D( pool_size=2,
                                           strides=2) )

vgg16_model.add(tf.keras.layers.Conv2D( filters=256, 
                                        kernel_size=3, 
                                        padding='same', 
                                        activation='relu') )

vgg16_model.add(tf.keras.layers.Conv2D( filters=256, 
                                        kernel_size=3, 
                                        padding='same', 
                                        activation='relu') )

vgg16_model.add(tf.keras.layers.Conv2D( filters=256, 
                                        kernel_size=3, 
                                        padding='same', 
                                        activation='relu') )

vgg16_model.add(tf.keras.layers.MaxPool2D( pool_size=2,
                                           strides=2) )

vgg16_model.add(tf.keras.layers.Conv2D( filters=512, 
                                        kernel_size=3, 
                                        padding='same', 
                                        activation='relu') )

vgg16_model.add(tf.keras.layers.Conv2D( filters=512, 
                                        kernel_size=3, 
                                        padding='same', 
                                        activation='relu') )

vgg16_model.add(tf.keras.layers.Conv2D( filters=512, 
                                        kernel_size=3, 
                                        padding='same', 
                                        activation='relu') )

vgg16_model.add(tf.keras.layers.MaxPool2D( pool_size=2,
                                           strides=2) )

vgg16_model.add(tf.keras.layers.Conv2D( filters=512, 
                                        kernel_size=3, 
                                        padding='same', 
                                        activation='relu') )

vgg16_model.add(tf.keras.layers.Conv2D( filters=512, 
                                        kernel_size=3, 
                                        padding='same', 
                                        activation='relu') )

vgg16_model.add(tf.keras.layers.Conv2D( filters=512, 
                                        kernel_size=3, 
                                        padding='same', 
                                        activation='relu') )

vgg16_model.add(tf.keras.layers.MaxPool2D( pool_size=2,
                                           strides=2,
                                           name='vgg16' ))

# Step 3 - Flattening
vgg16_model.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
vgg16_model.add(tf.keras.layers.Dense( units=256, 
                                       activation='relu',
                                       name='fac1' ))

vgg16_model.add(tf.keras.layers.Dense( units=128, 
                                       activation='relu',
                                       name='fac2' ))

# Step 5 - Output Layer
# vgg16_model.add(tf.keras.layers.Dense( units=1, 
#                                      activation='sigmoid' ))

vgg16_model.add(tf.keras.layers.Dense( units=2, 
                                       activation='softmax',
                                       name='output' ))

vgg16_model.summary()

# Part 3 - Training the vgg16
# Compiling the vgg16
# optimizion al gori thm is 'adam' means 'Adaptive Moment Estimation' 
# metrics measures the pcercentage of correct prediction a model makes.

# vgg16_model.compile( optimizer = 'adam', 
#                    loss = 'binary_crossentropy', 
#                    metrics = ['accuracy'])

opt = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)

vgg16_model.compile( optimizer = opt, 
                     loss = 'categorical_crossentropy', 
                     metrics = ['accuracy'])

# Training the vgg16 on the Training set and evaluating it on the Test set
vgg16_model.fit( x = training_set, 
                 validation_data = test_set, 
                 epochs = 15)

# vgg16_model.fit( x = training_set, 
#             validation_data = test_set, 
#             epochs = 25)
vgg16_model.save('vgg16_keras.keras')

# Conver models to onnx and export.

import tf2onnx
import onnxruntime
import onnx

spec = (tf.TensorSpec((None, 224, 224, 3), 
                       tf.float32, 
                       name="input"), )

output_path = "vgg16_onnx.keras.onnx"
model_onnx, _ = tf2onnx.convert.from_keras( model = vgg16_model, 
                                            input_signature = spec,
                                            opset = 13,
                                            output_path = output_path)

# python3 -m tf2onnx.convert --sved-model vgg16_onnx.keras --output vgg16_onnx.keras.onnx -opset 13
onnx.save_model(model_onnx, output_path)

# Part 4 - Making a single prediction
import numpy as np
from tensorflow.keras.preprocessing import image
# test_image = image.load_img( 'dataset/single_prediction/yell7.png', 
#                              target_size = (224, 224) )

test_image = image.load_img( 'dataset/single_prediction/yell8.png', 
                              target_size = (224, 224) )

test_image = image.img_to_array(test_image)
test_image = np.expand_dims( test_image, 
                             axis = 0)
result = vgg16_model.predict( test_image )
training_set.class_indices

print("[0][0]",result[0][0])
print("[0][1]",result[0][1])

# if result[0][0] == 1:
#     prediction = 'dog'
# else:
#     prediction = 'cat'
# print(prediction)

# save the models
# https://keras.io/guides/serialization_and_saving/
# https://onnxruntime.ai/docs/get-started/with-python.html
reconstructed_vgg16Model = tf.keras.models.load_model('vgg16_keras.keras')
new_result = reconstructed_vgg16Model.predict( test_image )

print("[0][0]",new_result[0][0])
print("[0][1]",new_result[0][1])

# if new_result[0][0] == 1:
#     prediction = 'dog'
# else:
#     prediction = 'cat'
# print(prediction)

print("hello")