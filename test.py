import cv2
import numpy as np

file = cv2.imread("/media/nico/Elements/Kitti Dataset/depth_selection/val_selection_cropped/groundtruth_depth/2011_09_26_drive_0002_sync_groundtruth_depth_0000000005_image_02.png",)

cv2.imshow("File",file)

file = np.array(file,dtype="f")

print(file[:3,:3,:])
print(file.shape)

cv2.waitKey(0)