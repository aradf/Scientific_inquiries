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
        self.loaded_cnnModel = tf.keras.models.load_model('cnn_keras.keras')
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
        print("[0][0]",result[0][0])
        print("[0][1]",result[0][1])
        print("[0][2]",result[0][2])
        print("[0][3]",result[0][3])
        print("[0][4]",result[0][4])
        print("[0][5]",result[0][5])
        print("[0][6]",result[0][6])
        print("[0][7]",result[0][7])
        print("[0][8]",result[0][8])
        print("[0][9]",result[0][9])

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
 
        input_name  = onnxruntime_inferenceSession.get_inputs()[0].name
        output_name = onnxruntime_inferenceSession.get_outputs()[0].name

        test_image = image.load_img( image_fileLocation, 
                                     target_size = (64, 64) )

        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims( test_image, 
                                     axis = 0)
       
        result = onnxruntime_inferenceSession.run(None, {'input': test_image})
        print("[0][0][0]",result[0][0][0])
        print("[0][0][1]",result[0][0][1])
        print("[0][0][2]",result[0][0][2])
        print("[0][0][3]",result[0][0][3])
        print("[0][0][4]",result[0][0][4])
        print("[0][0][5]",result[0][0][5])
        print("[0][0][6]",result[0][0][6])
        print("[0][0][7]",result[0][0][7])
        print("[0][0][8]",result[0][0][8])
        print("[0][0][9]",result[0][0][9])

# __name__
if __name__ == "__main__":

    cnn_model = ConvolutinalNeuralNetowork()
    cnn_model.predict( image_fileLocation = 'dataset/single_prediction/number-00.png' )
    cnn_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/number-00.png' )    
    cnn_model.predict( image_fileLocation = 'dataset/single_prediction/number-10.png' )
    cnn_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/number-10.png' )    
    cnn_model.predict( image_fileLocation = 'dataset/single_prediction/number-20.png' )    
    cnn_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/number-20.png' )    
    cnn_model.predict( image_fileLocation = 'dataset/single_prediction/number-30.png' )    
    cnn_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/number-30.png' )    
    cnn_model.predict( image_fileLocation = 'dataset/single_prediction/number-40.png' )    
    cnn_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/number-40.png' )    
    cnn_model.predict( image_fileLocation = 'dataset/single_prediction/number-50.png' )    
    cnn_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/number-50.png' )    
    cnn_model.predict( image_fileLocation = 'dataset/single_prediction/number-60.png' )    
    cnn_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/number-60.png' )    
    cnn_model.predict( image_fileLocation = 'dataset/single_prediction/number-70.png' )    
    cnn_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/number-70.png' )    
    cnn_model.predict( image_fileLocation = 'dataset/single_prediction/number-80.png' )    
    cnn_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/number-80.png' )    
    cnn_model.predict( image_fileLocation = 'dataset/single_prediction/number-90.png' )    
    cnn_model.predict_onnx( image_fileLocation = 'dataset/single_prediction/number-90.png' )    
    print("hello")

