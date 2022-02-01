import tensorflow as tf
import numpy as np
import tensorflow_transform as tft

def multiDepthLoss(groundTruth,depthsPredicted):
    
    depthsPredicted = tf.squeeze(depthsPredicted)
    
    valueMap = tf.where(tf.greater(groundTruth,0),1.,0.)
    
    loss = 0
    
    print(depthsPredicted.shape)
    
    for predSize in range(depthsPredicted.shape[0]):
        
        abs = tf.abs((groundTruth - depthsPredicted[predSize,:,:,:]))
        
        abs = valueMap*abs
        
        abs = tf.reduce_mean(abs,axis=0)
        abs = tf.reduce_mean(abs,axis=0)
        loss += tf.reduce_mean(abs,axis=0)
  
    return loss