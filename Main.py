import math
import cv2



preTrainedFaceData = cv2.CascadeClassifier('enter xml machine learning file') #<---- load pre-trained dataset on front facing self portraits 


#*** Choose image to detect a face inside
initImg = cv2.imread('filename.png/jpg')

#*** for opencv to use image correctly mustfirst convert to grayscale
gsImg = cv2.cvtColor(initImg, cv2.COLOR_BGR2GRAY) #<---- second parameter sets img background to gray

