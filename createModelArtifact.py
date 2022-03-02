import os
from progressbar import progressbar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import math
import tqdm
import cv2

class depthModel(tf.keras.Model):
    def __init__(self):
        super(depthModel, self).__init__()
        
        encoderKernels = [7,7,5,3,3]
        steps = len(encoderKernels)
        decoderKernels = [4,3,3]
        minChannels = 16
        
        self.encoders = []
        self.predictors = []
        self.upconvoluters = []
        self.reconvoluters = []
        self.upsamplingLayer = tf.keras.layers.UpSampling2D(size=(2,2))
        
        
        # The encoding layers
        for i, size in enumerate(encoderKernels):
            channels = minChannels*int(math.pow(2,i))
        
            self.encoders.append(tf.keras.layers.Conv2D(channels,(size, size), activation='relu',strides=2,padding="same",name=f"Encoder-{i}"))
           
        # Layers in between 
        self.predictors.append(tf.keras.layers.Conv2D(1, decoderKernels[2], activation='relu',padding="same",name="Predictor-0"))
        
        # The decoding layers
        for i in range(steps-1,0,-1):
            # Set channels down
            channels /= 2
            
            self.upconvoluters.append(tf.keras.layers.Conv2DTranspose(channels, decoderKernels[0], strides=2,activation='relu',padding="same",name=f"Upconvoluter-{i-steps+1}"))
            self.reconvoluters.append(tf.keras.layers.Conv2D(channels, decoderKernels[1], activation='relu',padding="same",name=f"ReConvoluter-{i-steps+1}"))
            self.predictors.append(tf.keras.layers.Conv2D(1, decoderKernels[2], activation='relu',padding="same",name=f"Predictor-{i-steps+2}"))
    
    def call(self,inputs):
        predictions = []
        convResults = [] # For saving the outputs of the encoding convolutional layers, outputs are needed for concatenation in corresponding later layer.
        
        inputs = tf.cast(inputs,tf.float16)
        
        x = inputs
        
        for encodingLayer in self.encoders:
            x = encodingLayer(x)
            convResults.append(x)
            
        predictions.append(self.predictors[0](x))
        
        
        for i,predictor in enumerate(self.predictors[1:]):
            
            x = self.upconvoluters[i](x)
            
            # Upsample lastPrediction, search for convX-output, add output of upconvLayer
            concatenation = tf.concat((self.upsamplingLayer(predictions[-1]),x,convResults[-2-i]),axis=3)
            
            x = self.reconvoluters[i](concatenation)
            
            predictions.append(predictor(x))
            
        predictions = self.scalePredictions(predictions)
        
        return predictions
        
    def scalePredictions(self,predictions,imSize=(352,480)):
    
        all = []

        for pred in predictions: 
            
            all.append(tf.image.resize(pred,imSize))
            
        return tf.stack(all)    

    def multiDepthLoss(self,groundTruth,depthsPredicted):
        
        depthsPredicted = tf.squeeze(depthsPredicted)
        
        print(f"depthsPredicted: {depthsPredicted[3,0,190:193,200:203].numpy()}")
        
        valueMap = tf.where(tf.greater(groundTruth,0),1.,0.)
        nonZeros = tf.math.count_nonzero(valueMap)
        
        loss = 0
        
        for predSize in range(depthsPredicted.shape[0]):
            abs = tf.abs((groundTruth - depthsPredicted[predSize,:,:,:]))
            
            abs = valueMap*abs

            loss += tf.reduce_sum(abs)
            
            # Mean enthält wahrscheinlich noch die vielen Nullen: tf.reduce_sum… / tf.reduce_sum(valueMap)

        return loss/tf.cast(nonZeros,dtype=tf.float32)/depthsPredicted.shape[0]

    def fit(self,x,y,batchSize=None,epochs=1):
        
        if not batchSize:
                batchSize = len(x)
                
    
        
        #progressBar = tqdm.tqdm(total=len(x))
        for epoch in range(epochs):
            
            i = 0
            
            while i < len(x):
                area = [i,i+batchSize]
                if i+batchSize > len(x):
                    area[1]=len(x)
                i += batchSize
                    
                currPictures = x[area[0]:area[1]]
                currDepth = y[area[0]:area[1]]
            
                with tf.GradientTape() as tape:
                
                    predictions = self.call(currPictures)
                    
                    
                
                    loss = self.multiDepthLoss(currDepth,predictions)
                    print(f"Loss: {loss}")
                    #progressBar.set_description(desc=f"Loss: {loss.numpy()}",refresh=True)
                    self.history = loss.numpy()
                    
                    grads = tape.gradient(loss,self.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads,self.trainable_weights))
                    
                #progressBar.update(i)
        #progressBar.close()
        
        cv2.imwrite("Predictions3.png",predictions[3,0,:,:].numpy())
        cv2.imwrite("Predictions3.png",predictions[4,0,:,:].numpy())
        cv2.imwrite("GroundTruth.png",y[0,:,:])

def createModel(learningRate,regularization_factor=0):
    
    model = depthModel()
    model.compile(
        optimizer = tf.keras.optimizers.SGD(learningRate),
        loss = model.multiDepthLoss
    )
    model.build((None,352,480,3))
    model.summary()
    
    return model