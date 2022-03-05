import numpy as np
import cv2

def rawToColorful(original,depthMap):

    
    originalHSV = cv2.cvtColor(original,cv2.COLOR_BGR2HSV)


    originalHSV[:,:,0] = exponential(depthMap[:,:,0],0.03)
    
    
    originalHSV[:,:,1] = np.ones(originalHSV.shape[:2])*255
    
    h = 100
    
    values = np.array(originalHSV[:,:,2],np.float32)
    
    # Linear
    originalHSV[:,:,2] = np.array(h + values* (255-h)/255,dtype=np.uint8)

    # Quadratic
    # a = (255 - h) /255/255
    # originalHSV[:,:,2] = np.array(h + values * values * a)

    ansicht = cv2.cvtColor(originalHSV,cv2.COLOR_HSV2BGR)
    
    return ansicht

def exponential(matrix,k):
    
    matrix = np.array(matrix,np.float32)
    
    matrix = np.exp(-k*matrix)*100
    
    matrix = np.array(matrix,np.uint8)
    
    return matrix
    
if __name__ == '__main__':
    
    print(exponential(1,0.013))


    original = cv2.imread("Predictions/media_images_prediction4_1_5bcb121203d7130ec0ef.png")
    depthMap = cv2.imread("Predictions/media_images_prediction4_351_8c2d772ee72cdd50c809.png")
    
    # original = np.ones((70,100,3),dtype=np.uint8)*200
    # depthMap = np.ones((70,100,3),dtype=np.uint8)
    
    # for i, row in enumerate(depthMap):
        
    #     depthMap[i,:,:] = depthMap[i,:,:] * i

    cv2.imshow("Original",rawToColorful(original,depthMap))
    cv2.imshow("DepthMap",depthMap)
    cv2.waitKey(0)