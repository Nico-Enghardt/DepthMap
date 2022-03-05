import os, sys
import platform
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import wandb,cv2
import tensorflow as tf
import numpy as np

import visualizeDepth, loss

run = wandb.init(job_type="model-evaluation")

modelName = "Dostojewski"

modelArtifact = run.use_artifact(modelName+":latest")
model_directory = modelArtifact.download()

model = tf.keras.models.load_model(model_directory,custom_objects={"multiDepthLoss":loss.multiDepthLoss})

def resize(img,destination=(352,480)):
    
    size = destination
    
    if img.shape[0]/img.shape[1] > 352/480:
        size = destination[1],int(img.shape[0]/img.shape[1]*destination[1])
        
    else:
        size = int(img.shape[1]/img.shape[0]*destination[0]),destination[0]

    return cv2.resize(img,size)

def evaluateModel():
    
    files = os.listdir("./TestPictures")
    
    for file in files:
        if file[:5] == "Depth":
            
            continue
        
        original = cv2.imread("./TestPictures/" + file)
        crop = resize(original)[-352:,:480]
        
        prediction = model.predict(np.expand_dims(crop,axis=0))[4,0]
        
        cv2.imshow("Output",cv2.resize(visualizeDepth.rawToColorful(crop,prediction),(480*2,352*2,)))
        cv2.waitKey(0)

if __name__ == '__main__':
    
    evaluateModel()