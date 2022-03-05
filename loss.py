import tensorflow as tf

def multiDepthLoss(groundTruth,depthsPredicted):
        
    depthsPredicted = tf.squeeze(depthsPredicted)
            
    valueMap = tf.where(tf.greater(groundTruth,0),1.,0.)
    nonZeros = tf.math.count_nonzero(valueMap)
    
    loss = 0
    
    for predSize in range(depthsPredicted.shape[0]):
        abs = tf.abs((groundTruth - depthsPredicted[predSize,:,:,:]))
        
        abs = valueMap*abs

        loss += tf.reduce_sum(abs)

    return loss/tf.cast(nonZeros,dtype=tf.float32)/depthsPredicted.shape[0]