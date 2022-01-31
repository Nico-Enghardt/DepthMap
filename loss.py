import tensorflow as tf
import numpy as np
import tensorflow_transform as tft

def multiDepthLoss(groundTruth,depthsPredicted):
    
    depthsPredicted = tf.squeeze(depthsPredicted)
    
    print(depthsPredicted.shape)
    print(groundTruth.shape)
    
    loss = 0
    
    for predSize in range(depthsPredicted.shape[0]):
        
        abs = tf.abs((groundTruth - depthsPredicted[predSize,:,:,:]))
        
        tf.reduce_mean(abs,axis=0)
        abs = tf.reduce_mean(abs,axis=0)
        loss += tf.reduce_mean(abs,axis=0)
        
        print(len(loss))
 
    return loss