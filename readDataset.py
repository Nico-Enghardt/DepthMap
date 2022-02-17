import os
import numpy as np
import cv2
from tqdm import tqdm

def readDatasetTraining(path,datasetSize=None,percentageDataset=0.8):
    
    print("Loading training pictures:")
    pictures = readFromFolder(path+"/val_selection_cropped/image",datasetSize,format=(1,352,480,3),)
    print("Loading depth data:")
    trueDepth = readFromFolder(path+"/val_selection_cropped/groundtruth_depth",datasetSize,format=(1,352,480))/256
    
    splitNumber = int(len(pictures)*percentageDataset)

    trainingPictures,trainingTrueDepth = pictures[:splitNumber,:], trueDepth[:splitNumber,:]
    testPictures, testTrueDepth = pictures[splitNumber:,:], trueDepth[splitNumber:,:]

    return trainingPictures,trainingTrueDepth,testPictures,testTrueDepth

def readFromFolder(path,datasetSize,format):
    files = os.listdir(path);
    
    if datasetSize:
        files = files[:datasetSize]
        
    list = []

    for file in tqdm(files):
        pathToFile = path+"/"+file
        if len(format)==3:
            list.append(cv2.imread(pathToFile)[:,368:848,0])
        else:
            list.append(cv2.imread(pathToFile)[:,368:848,:])

    #if shuffling: random.shuffle(pictures)
    
    return np.stack(list)  # Delete first row (random inintialisation of np.empty)
