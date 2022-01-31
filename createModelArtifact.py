import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import math
import numpy as np
import cv2
from loss import multiDepthLoss

def createModel(learning_rate,regularization_factor):
    

    encoderKernels = [7,7,5,3,3]

    steps = len(encoderKernels)

    decoderKernels = [4,3,3]

    minChannels = 16

    inputs = tf.keras.Input(shape=(352,480,3))
    predictions = [];

    convResults = []  # For saving the outputs of the encoding convolutional layers, outputs are needed for concatenation in corresponding later layer.

    for i, size in enumerate(encoderKernels):
        channels = minChannels*int(math.pow(2,i))
        
        newLayer = tf.keras.layers.Conv2D(channels, (size, size), activation='relu',strides=2,padding="same")

        if (i==0):
            convResults.append(newLayer(inputs))
            continue
        
        convResults.append(newLayer(convResults[-1]))
        

    smallestPrediction = tf.keras.layers.Conv2D(1, decoderKernels[2], activation='relu',padding="same")

    predictions.append(smallestPrediction(convResults[-1]))

    lastLayerOut = "just initialize"
    
    upsamplingLayer = tf.keras.layers.UpSampling2D(size=(2,2))


    for i in range(steps-1,0,-1):
        # Set channels down
        channels /= 2
        
        # Add Deconvolution-Layer: upconvX
        upconvLayer = tf.keras.layers.Conv2DTranspose(channels, decoderKernels[0], strides=2,activation='relu',padding="same")
        
        if i==steps-1:
            lastLayerOut = upconvLayer(convResults[-1])
        else: lastLayerOut = upconvLayer(lastLayerOut)
        
        # Add another Convolutional Layer: iconvX
        iconvLayer = tf.keras.layers.Conv2D(channels, decoderKernels[1], activation='relu',padding="same")
        
        # Upsample lastPrediction, search for convX-output, add output of upconvLayer
        
        
        
        concatenation = tf.concat((upsamplingLayer(predictions[-1]),lastLayerOut,convResults[i-1]),axis=3)
        
        lastLayerOut = iconvLayer(concatenation)
        
        # Add prediction layer: predX
        predLayer = tf.keras.layers.Conv2D(1, decoderKernels[2], activation='relu',padding="same")
            
        predictions.append(predLayer(lastLayerOut))
        
    predictions = scalePredictions(predictions)
        
        

    depthNetwork = tf.keras.Model(inputs=inputs,outputs = predictions,name="depthEstimationCNN")
    depthNetwork.compile(optimizer="Adam",loss=multiDepthLoss)
    depthNetwork.build(input_shape=tf.TensorShape((1,352,480,3)))


    depthNetwork.summary()

    tf.keras.utils.plot_model(depthNetwork,"depthNetwork.png",show_shapes=True,rankdir="TR")
    
    return depthNetwork

def scalePredictions(predictions):

    all = []

    for pred in predictions: 
        
        all.append(tf.image.resize(pred,(352,480)))
        
    return tf.stack(all)