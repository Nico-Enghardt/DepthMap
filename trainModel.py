import os, sys
import platform
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import wandb
import tensorflow as tf
import numpy as np
from readDataset import *
import createModelArtifact

local = False
if platform.node()=="kubuntu20nico2":
    local = True

modelName = None
architecture = "MayerN"
max_epochs = 3
batch_fraction = 0.21
learningRate = 0.0004

datasetSize = 1000
if local:
    datasetSize = 100

print(f"Using {datasetSize} picture-depth data-pairs for training and testing.")

run = wandb.init(job_type="model-training", config={"regularization":0,"architecture":architecture,"learningRate":0.00013})

# Load Model --------------------------------------------------------------------------------------------

if modelName:

    modelArtifact = run.use_artifact(modelName+":latest")
    model_directory = modelArtifact.download()

    #model = tf.keras.models.load_model(model_directory,custom_objects={"multiDepthLoss":multiDepthLoss})

else:
    modelName = "newModel"
    model = createModelArtifact.createModel(learningRate)

# Load and prepare Training Data -------------------------------------------------------------------------------------------------

datasetFolder = "/media/nico/Elements/Kitti Dataset/depth_selection"
if not local:
    datasetFolder = "/kaggle/input/kitti-depth-estimation-selection/depth_selection"

trainingPictures,trainingTrueDepth,testPictures,testTrueDepth = readDatasetTraining(datasetFolder,datasetSize)

run.config["trainingExamples"] = trainingPictures.shape[0]
run.config["testExamples"] = testPictures.shape[0]

# Fit model to training data --------------------------------------------------------------------------------------------

nth = 0

cv2.imwrite("Predictions/GroundTruth.png",testTrueDepth[nth,:,:])

wandb.log({"groundTruth":wandb.Image("Predictions/GroundTruth.png")},commit=False)

batchSize = int(batch_fraction*trainingPictures.shape[0])
print(f"Using batchsize: {batchSize}")
run.config["batchSize"] = batchSize;

for e in range(max_epochs):
    print("Epoch: "+ str(e))
    
    model.fit(trainingPictures,trainingTrueDepth,batchSize,epochs=1)
    wandb.log({"loss":model.history})
    
    if (e%10==5):
        predictions = model.predict(testPictures)
        
        print(f"Current Predictions: {predictions[4,0,190:195,200]}")
    
    if (e%50==0):
        
        predictions = model.predict(testPictures)
        
        cv2.imwrite("Predictions/Predictions4.png",cv2.resize(predictions[4,nth,:,:],(480*2,352*2)))
        
        wandb.log({"prediction4":wandb.Image("Predictions/Predictions4.png")},commit=False)
    

# Save Model online and finish run –------------------------------------------------------------------------------------

model.save("artifacts/"+model.name)

modelArtifact = wandb.Artifact(model.name,type="model")
modelArtifact.add_dir("artifacts/"+model.name)

run.log_artifact(modelArtifact)

#Test model's predictions
evalLoss = model.evaluate(testPictures)
print(f"\n Loss on testSet: {evalLoss}")
wandb.log({"evalLoss":evalLoss})

wandb.finish()

