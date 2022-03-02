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
max_epochs = 400
batch_fraction = 0.2
learningRate = 0.0001

if len(sys.argv)>1:
    datasetSize = int(sys.argv[1])
else:
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
    
    if (e%50==0):
        
        predictions = model.predict(testPictures)
        
        cv2.imwrite("Predictions/Predictions4.png",cv2.resize(predictions[4,nth,:,:].numpy(),(480*2,352*2)))
        
        wandb.log({"prediction4":wandb.Image("Predictions/Predictions4.png")},commit=False)
    
    
#Test model's predictions
predictions = model.predict(testPictures[1,:,:,:])
print("\n Predictions:")
#cv2.imshow("Prediction",predictions[-1,0,:,:,:])
print(predictions)
print("\n")

# Save Model online and finish run –------------------------------------------------------------------------------------

model.save("artifacts/"+modelName)

modelArtifact = wandb.Artifact(modelName,type="model")
modelArtifact.add_dir("artifacts/"+modelName)

run.log_artifact(modelArtifact)

wandb.finish()

