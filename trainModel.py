import os
import platform
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import wandb
import tensorflow as tf
import numpy as np
from readDataset import *
from loss import multiDepthLoss
import createModelArtifact

local = False
if platform.node()=="kubuntu20nico2":
    local = True

modelName = None
architecture = "to be named"
max_epochs = 3
batch_fraction = 1/2
regularization_factor =  0.5
learning_rate = 0.000001
percentageDataset = 1;

#run = wandb.init(job_type="model-training", config={"epochs":0,"learning_rate":learning_rate,"batch_fraction":batch_fraction,"regularization":regularization_factor,"architecture":architecture})

# Load Model --------------------------------------------------------------------------------------------

if modelName:

    modelArtifact = run.use_artifact(modelName+":latest")
    model_directory = modelArtifact.download()

    model = tf.keras.models.load_model(model_directory,custom_objects={"multiDepthLoss":multiDepthLoss})

else:
    model = createModelArtifact.createModel(learning_rate,regularization_factor)

# Load and prepare Training Data -------------------------------------------------------------------------------------------------

datasetFolder = "/media/nico/Elements/Kitti Dataset/depth_selection"
if not local:
    datasetFolder = "/kaggle/input/kitti-depth-estimation-selection/depth_selection"

trainingPictures,trainingTrueDepth,testPictures,testTrueDepth = readDatasetTraining(datasetFolder)

#run.config["trainingExamples"] = trainingLabels.shape[0];
#run.config["testExamples"] = testLabels.shape[0];

# Fit model to training data --------------------------------------------------------------------------------------------

e = 0
batch_size = int(batch_fraction*trainingPictures.shape[0])
#run.config["batch-size"] = batch_size;

while e < max_epochs:
    print("Epoch: "+ str(e))
    model.fit(x=trainingPictures,y=trainingTrueDepth,batch_size=batch_size,verbose=1)

    # i = 0
    # metrics = []

    # while i < len(trainingPictures):
    #     range = [i,i+batch_size]
        
    #     if i+batch_size > len(trainingPictures):
    #         range[1]=len(trainingPictures)
    #     currPictures = np.array(trainingPictures[range[0]:range[1]])
    #     currLabels = np.array(trainingLabels[range[0]:range[1]])

    #     i=i+batch_size
    #     model.fit(x=currPictures,y=currLabels,verbose=1)
        
    #     currMetrics = model.history.history

    #     metrics.append([currMetrics["loss"][0],currMetrics["loss3D"][0],currMetrics["heightError"][0],currMetrics["planeError"][0]])

    # metrics = np.average(metrics,axis=0)
    # wandb.log({"loss":metrics[0],"acc3D":metrics[1],"heightError":metrics[2],"planeError":metrics[3]})

    # if (e % 5 == 0):
    #     metrics = model.evaluate(x=testPictures,y=testLabels,batch_size=batch_size,verbose=2)
    #     wandb.log({"testLoss":metrics[0],"testAcc3D":metrics[1],"testHeightError":metrics[2],"testPlaneError":metrics[3]},commit=False)
        
    e = e+1
    


model.summary()

# Test model's predictions
predictions = model.predict(testPictures[1,:,:,:])
print("\n Predictions:")
print(predictions)
print("\n")

# Save Model online and finish run –------------------------------------------------------------------------------------

modelName = "newModel"

model.save("artifacts/"+modelName)

modelArtifact = wandb.Artifact(modelName,type="model")
modelArtifact.add_dir("artifacts/"+modelName)

#run.log_artifact(modelArtifact)

#wandb.finish()

