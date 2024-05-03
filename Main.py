import math
import cv2


#*** pre made dataset import
#preTrainedFaceData = cv2.CascadeClassifier('enter xml machine learning file') #<---- load pre-trained dataset on front facing self portraits 


#*** choose image to detect a face inside
initImg = cv2.imread('RedCarpetFace1.jpg')

#*** for opencv to use image correctly mustfirst convert to grayscale
gsImg = cv2.cvtColor(initImg, cv2.COLOR_BGR2GRAY) #<---- second parameter sets img background to gray

#*** display in image view grayscaled image
cv2.imshow('AI Face detector', gsImg ) #<--- parameter 1 is AppBar title and 2 is Image to Open
cv2.waitKey() #<--- waitKey to prevent photo viewer from instantly exiting