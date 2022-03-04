import numpy as np
import cv2

def rawToColorful(original,depthMap):

    
    originalHSV = cv2.cvtColor(original,cv2.COLOR_BGR2HSV)

    hue = depthMap[:,:,0]
    originalHSV[:,:,0] = hue
    originalHSV[:,:,1] = np.ones(hue.shape)*255

    ansicht = cv2.cvtColor(originalHSV,cv2.COLOR_HSV2BGR)


    return ansicht

if __name__ == '__main__':


    original = cv2.imread("Predictions/original.png")
    depthMap = cv2.imread("Predictions/prediction4.png")

    cv2.imshow("Original",rawToColorful(original,depthMap))
    cv2.waitKey(0)