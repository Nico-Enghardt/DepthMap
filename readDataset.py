import os
import numpy as np
import cv2
from tqdm import tqdm

def readDatasetTraining(path,datasetSize=None,percentageDataset=0.99):
    
    print("Loading training pictures:")
    pictures = readFromFolderChop(path+"/val_selection_cropped/image",datasetSize,format="image",)
    print("Loading depth data:")
    trueDepth = readFromFolderChop(path+"/val_selection_cropped/groundtruth_depth",datasetSize,format="groundTruth")
    
    splitNumber = int(len(pictures)*percentageDataset)

    trainingPictures,trainingTrueDepth = pictures[:splitNumber,:], trueDepth[:splitNumber,:]
    testPictures, testTrueDepth = pictures[splitNumber:,:], trueDepth[splitNumber:,:]

    return trainingPictures,trainingTrueDepth,testPictures,testTrueDepth

def readFromFolder(path,datasetSize,format):
    files = os.listdir(path)
    
    if datasetSize:
        files = files[:datasetSize]
        
    list = []

    for file in tqdm(files):
        pathToFile = path+"/"+file
        if format=="groundTruth":
            list.append(cv2.imread(pathToFile)[:,368:848,0])
        else:
            list.append(cv2.imread(pathToFile)[:,368:848,:])
    
    return np.stack(list)

def readFromFolderChop(path,datasetSize,format):
    
    files = os.listdir(path)
    
    if datasetSize:
        files = files[:int(datasetSize/6)]
    
    list = []
    
    for file in tqdm(files):
        pathToFile = path + "/" + file
        if format=="groundTruth":
            img = cv2.imread(pathToFile)[142:334,:,0]
        else: img = cv2.imread(pathToFile)[142:334,:,:]
            
        list.extend((img[:,32:224],img[:,224:416],img[:,416:608],img[:,608:800],img[:,800:992],img[:,992:1184]))
    
    return np.stack(list)