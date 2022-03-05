import os, sys
import platform
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import wandb
import tensorflow as tf
import numpy as np

from readDataset import *
import visualizeDepth, createModelArtifact, loss

local = False
if platform.node()=="kubuntu20nico2":
    local = True

modelName = None
architecture = "MayerN"
max_epochs = 602
batch_fraction = 0.101
learningRate = 0.0004

datasetSize = 1000
if local:
    datasetSize = 600

print(f"Using {datasetSize} picture-depth data-pairs for training and testing.")

run = wandb.init(job_type="model-training", config={"architecture":architecture,"datasetType":"smallSquares6"})

# Load Model --------------------------------------------------------------------------------------------

if modelName:

    modelArtifact = run.use_artifact(modelName+":latest")
    model_directory = modelArtifact.download()
    print(f"Use saved Model: {modelName}:latest in dir: {model_directory}")
    model = tf.keras.models.load_model(model_directory,custom_objects={"multiDepthLoss":loss.multiDepthLoss})
    model.fit = createModelArtifact.depthModel.fit
    model.optimizer = tf.keras.optimizers.Adam()
    reloaded = True
    
    run.config["optimizer"] = "Adam"
    
    alreadyTrainedSteps = modelArtifact.metadata["trainedSteps"]
    

else:
    model = createModelArtifact.createModel()
    
    reloaded = False
    run.config["optimizer"] = "Adam"



# Load and prepare Training Data -------------------------------------------------------------------------------------------------

datasetFolder = "/media/nico/Elements/Kitti Dataset/depth_selection"
if not local:
    datasetFolder = "/kaggle/input/kitti-depth-estimation-selection/depth_selection"

trainingPictures,trainingTrueDepth,testPictures,testTrueDepth = readDatasetTraining(datasetFolder,datasetSize)

run.config["trainingExamples"] = trainingPictures.shape[0]
run.config["testExamples"] = testPictures.shape[0]

# Rotate Function

def randRotateDataset(pictures,groundTruth):
    
    print(" I just scrambled up your dataset a little =)")
    
    rotateInstructions = np.random.randint(0,4,pictures.shape[0])
    
    for i,r in enumerate(rotateInstructions):
        
        if r == 0:
            pictures[i] = np.rot90(pictures[i])
            groundTruth[i] = np.rot90(groundTruth[i])
        if r == 1:
            pictures[i] = np.flipud(pictures[i])
            groundTruth[i] = np.flipud(groundTruth[i])
        if r == 2:
            pictures[i] = np.fliplr(np.rot90(pictures[i]))
            groundTruth[i] = np.fliplr(np.rot90(groundTruth[i]))
    
    return pictures,groundTruth

# Fit model to training data --------------------------------------------------------------------------------------------

nth = 0

cv2.imwrite("Predictions/GroundTruth.png",testTrueDepth[nth,:,:])
cv2.imwrite("Predictions/original.png",testPictures[nth,:,:])

wandb.log({"groundTruth":wandb.Image("Predictions/GroundTruth.png")},commit=False)
wandb.log({"original":wandb.Image("Predictions/original.png")},commit=False)

batchSize = int(batch_fraction*trainingPictures.shape[0])
print(f"Using batchsize: {batchSize}")
run.config["batchSize"] = batchSize;

for e in range(max_epochs):
    print("Epoch: "+ str(e))
    
    step = e
    
    if reloaded:
        model.fit(model,trainingPictures,trainingTrueDepth,batchSize)
        step += alreadyTrainedSteps
        
    else:
        model.fit(trainingPictures,trainingTrueDepth,batchSize)
    wandb.log({"loss":model.history,"step":step})
    
    if (e%10==0):
        #predictions = model.predict(testPictures)
        #print(f"Current Predictions: {predictions[4,0,190:195,200].squeeze()}")
    
        trainingPictures,trainingTrueDepth = randRotateDataset(trainingPictures,trainingTrueDepth)
    
    
    if (e%50==0):
        
        predictions = model.predict(testPictures)
        
        visualizer0 = visualizeDepth.rawToColorful(testPictures[nth,:,:,:],predictions[4,nth,:,:])
        visualizer1 = visualizeDepth.rawToColorful(testPictures[nth+1,:,:,:],predictions[4,nth+1,:,:])
        visualizer2 = visualizeDepth.rawToColorful(testPictures[nth+2,:,:,:],predictions[4,nth+2,:,:])
        visualizer3 = visualizeDepth.rawToColorful(testPictures[nth+3,:,:,:],predictions[4,nth+3,:,:])
        visualizer4 = visualizeDepth.rawToColorful(testPictures[nth+4,:,:,:],predictions[4,nth+4,:,:])
        
        visualizer = np.concatenate((visualizer0,visualizer1,visualizer2,visualizer3,visualizer4),axis=1)
        
        cv2.imwrite("Predictions/Predictions4.png",cv2.resize(visualizer,(192*6*2,192*2)))
        
        wandb.log({"prediction4":wandb.Image("Predictions/Predictions4.png")},commit=False)
    

#Test model's predictions
evalLoss = model.evaluate(testPictures,testTrueDepth)
print(f"\n Loss on testSet: {evalLoss}")
wandb.log({"evalLoss":evalLoss})

# Save Model online and finish run –------------------------------------------------------------------------------------

model.save("artifacts/"+model.name)

trainedModelArtifact = wandb.Artifact(model.name,type="model",metadata={"trainedSteps":max_epochs,"evalLoss":evalLoss,"trainLoss":model.history,"inputShape":trainingPictures.shape[1:]})

if reloaded:
    trainedModelArtifact.metadata["trainedSteps"] += modelArtifact.metadata["trainedSteps"]

trainedModelArtifact.add_dir("artifacts/"+model.name)

run.log_artifact(trainedModelArtifact)



wandb.finish()

