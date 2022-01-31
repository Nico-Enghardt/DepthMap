import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import wandb
import numpy as np
import cv2
import tensorflow as tf
import loss
from readDataset import *

def drawCross(image,position,color=(237, 74, 28)):
    cv2.line(image,(position[0]-10,position[1]-10),(position[0]+10,position[1]+10),color)
    cv2.line(image,(position[0]+10,position[1]-10),(position[0]-10,position[1]+10),color)
    return image


# Different versions: 1. Saving to WandB, 2. Only look at dataset (not Model), 3. Show locally, slow and fast

modelName = "radiantFirework"
datasetName = "Huegray160"

run = wandb.init(job_type = "performance-evaluation",
    project="PointRecognition",
    )

modelArtifact = run.use_artifact(modelName+":latest")
model_directory = modelArtifact.download()
model = tf.keras.models.load_model(model_directory,custom_objects={"loss3D":loss.loss3D,"heightError":loss.heightError,"planeError":loss.planeError})

datasetFolder = "Datasets/" + datasetName + "/Testing"

files = os.listdir(datasetFolder);

collection = []
os.mkdir("Datasets/Temp")

for file in files:
        
    # Features to record:
    # modelName and datasetName
    # Performance Measures
    # all three Videos
    
    if file[-3:] == "npy": continue
    else: file = file[:-3]    
    
    testPictures = loadVideo(datasetFolder+"/"+file+"mp4",flatten=False)
    testLabels = loadNumpy(datasetFolder+"/"+file+"npy")

    shape = testPictures.shape
    testPicturesFlat = np.reshape(testPictures,(shape[0],shape[1]*shape[2]*shape[3]))
    
    # Evaluate Model
    
    metrics = model.evaluate(x=testPicturesFlat,y=testLabels)    
    
    # Video reformation and tag data to video
    
    predictions = model.predict(testPicturesFlat)
    predictions = np.array(predictions,dtype='int')
    
    video = []
    
    for i in range(testPictures.shape[0]): # Jedes Bild im Video bearbeiten
        
        # Reconstruct original image
        
        hue = testPictures[i,:,:,0]
        gray = testPictures[i,:,:,1]
        third = np.ones(hue.shape,dtype="float32") * 100
        
        combination = np.stack((hue,gray,third),axis=2)
        
        frame = cv2.cvtColor(combination,cv2.COLOR_HLS2RGB)
        
        w,h = 160*2,120*2
        
        frame = cv2.resize(frame,(w,h))
        
        # Tag data to Video
        
        truePosition = np.array(testLabels[i,:],"int32")
        predPosition = np.array(predictions[i,:],"int32")
        
        trueCross = truePosition[0:2] * 2
        predCross = predPosition[0:2] * 2
        
        frame = drawCross(frame,trueCross)
        frame = drawCross(frame,predCross,color=(0,0,0))
                          
        video.append(frame)
        
        cv2.imshow("Frame",frame)
        cv2.waitKey(3)
        
    video = np.array(video)
    skvideo.io.vwrite("Datasets/Temp/Temp"+str(i)+".mp4",video)
    
    wandbVideo = wandb.Video("Datasets/Temp/Temp"+str(i)+".mp4")

    collection.append([metrics[1],metrics[2],metrics[3],wandbVideo,modelName,datasetName])
    print(collection)
       
    
videoDisplay = wandb.Table(columns=["testAcc3D","testHeightError","testPlaneError","resultVideo","modelName","datasetName"],data=collection)
run.log({"VideoDisplay":videoDisplay})

wandb.finish()

os.rmdir("Datasets/Temp",)