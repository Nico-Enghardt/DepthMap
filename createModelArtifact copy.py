import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import math
import numpy as np
import cv2
from loss import multiDepthLoss

def createModel(learning_rate,regularization_factor=0):
    
    

    

    inputs = tf.keras.Input(shape=(352,480,3))
    predictions = [];

    convResults = []  

    if (i==0):
            convResults.append(newLayer(inputs))
            continue
        
        convResults.append(newLayer(convResults[-1]))
        

    

    predictions.append(smallestPrediction(convResults[-1]))

    lastLayerOut = "just initialize"
    
    


    for i in range(steps-1,0,-1):
        # Set channels down
        
        # Add Deconvolution-Layer: upconvX

        if i==steps-1:
            lastLayerOut = upconvLayer(convResults[-1])
        else: lastLayerOut = upconvLayer(lastLayerOut)
        
        # Add another Convolutional Layer: iconvX
        
        
        
        
        concatenation = tf.concat((upsamplingLayer(predictions[-1]),lastLayerOut,convResults[i-1]),axis=3)
        
        lastLayerOut = iconvLayer(concatenation)
        
        # Add prediction layer: predX            
        predictions.append(predLayer(lastLayerOut))
        
    predictions = scalePredictions(predictions)
    
        

    depthNetwork = tf.keras.Model(inputs=inputs,outputs = predictions,name="depthEstimationCNN")
    depthNetwork.compile(optimizer="Adam",loss=multiDepthLoss)
    depthNetwork.build(input_shape=tf.TensorShape((1,352,480,3)))


    depthNetwork.summary()

    tf.keras.utils.plot_model(depthNetwork,"depthNetwork.png",show_shapes=True,rankdir="TR")
    
    return depthNetwork


