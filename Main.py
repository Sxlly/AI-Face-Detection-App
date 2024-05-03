import math
import cv2
from random import *


#*** pre made dataset import
preTrainedFaceData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #<---- load pre-trained dataset on front facing self portraits 


#*** choose image to detect face(s) inside
initImg = cv2.imread('RedCarpetFace1.jpg')
#*** choose a video to detect face(s) inside
initVideo = cv2.VideoCapture("David Guetta & OneRepublic - I Don't Wanna Wait (Official Video) (720p).mp4")


#*** for opencv to use image correctly mustfirst convert to grayscale
gsImg = cv2.cvtColor(initImg, cv2.COLOR_BGR2GRAY) #<---- second parameter sets img through RGB to gray

#*** Detect Facial features within image
faceCoordinates = preTrainedFaceData.detectMultiScale(gsImg) #<--- the classifier trained the "preTrainedFaceData" variable to detect faces accuratley due to funneling it multiple positive and negative data sets (imgs with faces and without)
                                    #^^^ multiscale method allows images to be various dimensions (scales)
print(faceCoordinates)#<--- prints coordinates of facial borders to console (terminal)


#*** Draw Rectangle around detected face
for (x, y, width, height) in faceCoordinates: #<--- Loop through full list of coordinates and draw a rectangle around face dimensions all times a face is detected
    cv2.rectangle(initImg, (x, y), (x+width,y+height), (0, 255, 0), 3)  #<--- (img, (x, y), (x+w, y+w), (B, G, R), BorderWidth)
                                                        #^^^ RGB value for pure green colour


#*** infinite loop to run continuosly over mp4 video
while True:

    posFrameRead, currentFrame = initVideo.read() #<--- Read current video frames (boolean T/F, current video frame value)
    grayscaleFrame = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY) #<---- second parameter sets img through RGB to gray

    #*** Draw Rectangle around detected face
    for (x, y, width, height) in faceCoordinates: #<--- Loop through full list of coordinates and draw a rectangle around face dimensions all times a face is detected
        cv2.rectangle(currentFrame, (x, y), (x+width,y+height), (randrange(256), randrange(256), randrange(256)),3)  #<--- (img, (x, y), (x+w, y+w), (B, G, R), BorderWidth)
                                                             #^^^ RGB value for random colours




#*** display in image view grayscaled image
cv2.imshow('AI Face detector', initImg) #<--- parameter 1 is AppBar title and 2 is Image to Open
cv2.waitKey() #<--- waitKey to prevent photo viewer from instantly exiting