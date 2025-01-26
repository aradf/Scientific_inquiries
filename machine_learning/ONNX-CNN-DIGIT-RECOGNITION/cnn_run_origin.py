# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

import onnxruntime
import onnx

print( tf.__version__ ) 

class ConvolutinalNeuralNetowork:
    def __init__(self):
        # load the models
        # https://keras.io/guides/serialization_and_saving/
        # https://onnxruntime.ai/docs/get-started/with-python.html
        # https://onnxruntime.ai/docs/api/python/api_summary.html
        self.loaded_cnnModel = tf.keras.models.load_model('my_cnn.keras')
        self.model_onnx = onnx.load('cnn_onnx.keras.onnx')

    def predict(self, image_fileLocation=None):
        # Part 4 - Making a single prediction

        if image_fileLocation==None:
            return
        
        print(image_fileLocation)
        test_image = image.load_img( image_fileLocation, 
                                     target_size = (64, 64) )

        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims( test_image, 
                                    axis = 0)

        result = self.loaded_cnnModel.predict( test_image )
        if result[0][0] == 1:
            prediction = 'dog'
        else:
            prediction = 'cat'
        print(prediction)

    def predict_onnx(self, image_fileLocation=None):
        # Part 4 - Making a single prediction
        if image_fileLocation==None:
            return

        output_path = 'cnn_onnx.keras.onnx'
        # model_onnx = onnx.load('cnn_onnx.keras.onnx')
        output_names = [n.name for n in self.model_onnx.graph.output]
        providers = ['CPUExecutionProvider']
        onnxruntime_inferenceSession = onnxruntime.InferenceSession( output_path, 
                                                                     providers = providers)

        test_image = image.load_img( image_fileLocation, 
                                     target_size = (64, 64) )

        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims( test_image, 
                                    axis = 0)
       
        result = onnxruntime_inferenceSession.run(None, {'input': test_image})
        
        if result[0][0] == 1:
            prediction = 'dog'
        else:
            prediction = 'cat'
        print(prediction)

# __name__
if __name__ == "__main__":

    cnn_model = ConvolutinalNeuralNetowork()
    cnn_model.predict( image_fileLocation = 'dataset/single_prediction/cat_or_dog_1.jpg' )
    cnn_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/cat_or_dog_1.jpg' )    
    cnn_model.predict( image_fileLocation = 'dataset/single_prediction/cat_or_dog_2.jpg' )
    cnn_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/cat_or_dog_2.jpg' )    
    cnn_model.predict( image_fileLocation = 'dataset/single_prediction/cat_or_dog_3.jpg' )    
    cnn_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/cat_or_dog_3.jpg' )    
    cnn_model.predict( image_fileLocation = 'dataset/single_prediction/cat_or_dog_4.jpg' )    
    cnn_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/cat_or_dog_4.jpg' )    
    cnn_model.predict( image_fileLocation = 'dataset/single_prediction/cat_or_dog_5.jpg' )    
    cnn_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/cat_or_dog_5.jpg' )    
    cnn_model.predict( image_fileLocation = 'dataset/single_prediction/cat_or_dog_6.jpg' )    
    cnn_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/cat_or_dog_6.jpg' )    
    cnn_model.predict( image_fileLocation = 'dataset/single_prediction/cat_or_dog_7.jpg' )    
    cnn_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/cat_or_dog_7.jpg' )    
    cnn_model.predict( image_fileLocation = 'dataset/single_prediction/cat_or_dog_8.jpg' )    
    cnn_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/cat_or_dog_8.jpg' )    
    cnn_model.predict( image_fileLocation = 'dataset/single_prediction/cat_or_dog_9.jpg' )    
    cnn_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/cat_or_dog_9.jpg' )    
    print("hello")

