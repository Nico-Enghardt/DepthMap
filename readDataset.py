import os
import numpy as np
import cv2

def readDatasetTraining(path,shuffleMode="shuffleBatches",percentageDataset=0.8,onlyFile=False):
    
    pictures = readFromFolder(path+"/val_selection_cropped/image",format=(1,352,480,3))
    trueDepth = readFromFolder(path+"/val_selection_cropped/groundtruth_depth",format=(1,352,480))/256
    
    splitPercentage = int(len(pictures)*percentageDataset)

    trainingPictures,trainingTrueDepth = pictures[:splitPercentage,:], trueDepth[:splitPercentage,:]
    testPictures, testTrueDepth = pictures[splitPercentage:,:], trueDepth[splitPercentage:,:]

    return trainingPictures,trainingTrueDepth,testPictures,testTrueDepth

def readFromFolder(path,format):
    files = os.listdir(path);
    
    files = files[:500]
        
    list = np.empty(format)

    for num,file in enumerate(files):
        print(num)
        pathToFile = path+"/"+file
        if len(format)==3:
            list = np.concatenate((list,np.expand_dims(cv2.imread(pathToFile)[:,368:848,0],axis=0)))
        else:
            list = np.concatenate((list,np.expand_dims(cv2.imread(pathToFile)[:,368:848,:],axis=0)))


    #if shuffling: random.shuffle(pictures)
    
    return list[1:,:,:]  # Delete first row (random inintialisation of np.empty)
