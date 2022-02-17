import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import math

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
        
            self.encoders.append(tf.keras.layers.Conv2D(channels,(size, size), activation='relu',strides=2,padding="same",name=f"Encoder #{i}"))
           
        # Layers in between 
        self.predictors.append(tf.keras.layers.Conv2D(1, decoderKernels[2], activation='relu',padding="same",name="Predictor #0"))
        
        # The decoding layers
        for i in range(steps-1,0,-1):
            # Set channels down
            channels /= 2
            
            self.upconvoluters.append(tf.keras.layers.Conv2DTranspose(channels, decoderKernels[0], strides=2,activation='relu',padding="same",name=f"Upconvoluter #{i-steps+1}"))
            self.reconvoluters.append(tf.keras.layers.Conv2D(channels, decoderKernels[1], activation='relu',padding="same",name=f"Reconvoluter #{i-steps+1}"))
            self.predictors.append(tf.keras.layers.Conv2D(1, decoderKernels[2], activation='relu',padding="same",name=f"Predictor #{i-steps+1}"))
    
    def call(self,inputs):
        predictions = []
        convResults = [] # For saving the outputs of the encoding convolutional layers, outputs are needed for concatenation in corresponding later layer.
        
        inputs = tf.cast(inputs,tf.float16)
        
        x = self.encoders[0](inputs)
        convResults.append(x)
        
        for encodingLayer in self.encoders[1:]:
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
        print(predictions.shape)
        
        return predictions
        
    def scalePredictions(self,predictions,imSize=(352,480)):
    
        all = []

        for pred in predictions: 
            
            all.append(tf.image.resize(pred,imSize))
            
        return tf.stack(all)    

    def multiDepthLoss(self,groundTruth,depthsPredicted):
        
        depthsPredicted = tf.squeeze(depthsPredicted)
        
        valueMap = tf.where(tf.greater(groundTruth,0),1.,0.)
        
        loss = 0
        
        for predSize in range(depthsPredicted.shape[0]):
            abs = tf.abs((groundTruth - depthsPredicted[predSize,:,:,:]))
            
            abs = valueMap*abs
            
            abs = tf.reduce_mean(abs,axis=0)
            abs = tf.reduce_mean(abs,axis=0)
            loss += tf.reduce_mean(abs,axis=0)
            
            # Mean enthält wahrscheinlich noch die vielen Nullen: tf.reduce_sum… / tf.reduce_sum(valueMap)

        return loss

    def fit(self,x,y,epochs=1):
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                
                predictions = self.call(x)
                
                loss = self.multiDepthLoss(y,predictions)
                print(loss)
                
                grads = tape.gradient(loss,self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads,self.trainable_weights))
                

            
        

def createModel(learning_rate,regularization_factor=0):
    
    model = depthModel()
    model.compile(
        optimizer = tf.keras.optimizers.Adam(),
        loss = model.multiDepthLoss
    )
    
    return model