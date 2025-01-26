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
        self.loaded_vgg16Model = tf.keras.models.load_model('vgg16_keras.keras')
        self.model_onnx = onnx.load('vgg16_onnx.keras.onnx')

    def predict(self, image_fileLocation=None):
        # Part 4 - Making a single prediction

        if image_fileLocation==None:
            return
        
        print(image_fileLocation)
        test_image = image.load_img( image_fileLocation, 
                                     target_size = (224, 224) )

        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims( test_image, 
                                    axis = 0)

        result = self.loaded_vgg16Model.predict( test_image )
        print("[0][0]",result[0][0])
        print("[0][1]",result[0][1])

    def predict_onnx(self, image_fileLocation=None):
        # Part 4 - Making a single prediction
        if image_fileLocation==None:
            return

        output_path = 'vgg16_onnx.keras.onnx'
        # model_onnx = onnx.load('vgg16_onnx.keras.onnx')
        output_names = [n.name for n in self.model_onnx.graph.output]
        providers = ['CPUExecutionProvider']
        onnxruntime_inferenceSession = onnxruntime.InferenceSession( output_path, 
                                                                     providers = providers)
 
        input_name  = onnxruntime_inferenceSession.get_inputs()[0].name
        output_name = onnxruntime_inferenceSession.get_outputs()[0].name

        test_image = image.load_img( image_fileLocation, 
                                     target_size = (224, 224) )

        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims( test_image, 
                                     axis = 0)
       
        result = onnxruntime_inferenceSession.run(None, {'input': test_image})
        print("[0][0][0]",result[0][0][0])
        print("[0][0][1]",result[0][0][1])

# __name__
if __name__ == "__main__":

    vgg16_model = ConvolutinalNeuralNetowork()
    vgg16_model.predict( image_fileLocation = 'dataset/single_prediction/0.jpg' )
    vgg16_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/0.jpg' )    
    vgg16_model.predict( image_fileLocation = 'dataset/single_prediction/0_0_aidai_0014.jpg' )
    vgg16_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/0_0_aidai_0014.jpg' )    
    vgg16_model.predict( image_fileLocation = 'dataset/single_prediction/0_0_aidai_0074.jpg' )    
    vgg16_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/0_0_aidai_0074.jpg' )    
    vgg16_model.predict( image_fileLocation = 'dataset/single_prediction/yell1.jpg' )    
    vgg16_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/yell1.jpg' )    
    vgg16_model.predict( image_fileLocation = 'dataset/single_prediction/yell5.png' )    
    vgg16_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/yell5.png' )    
    vgg16_model.predict( image_fileLocation = 'dataset/single_prediction/yell8.png' )    
    vgg16_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/yell8.png' )    
    vgg16_model.predict( image_fileLocation = 'dataset/single_prediction/yell9.png' )    
    vgg16_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/yell9.png' )    
    vgg16_model.predict( image_fileLocation = 'dataset/single_prediction/yell10.png' )    
    vgg16_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/yell10.png' )    
    vgg16_model.predict( image_fileLocation = 'dataset/single_prediction/yell11.png' )    
    vgg16_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/yell11.png' )    
    print("hello")

