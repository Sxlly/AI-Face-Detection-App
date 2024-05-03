import math
import cv2


#*** pre made dataset import
preTrainedFaceData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #<---- load pre-trained dataset on front facing self portraits 


#*** choose image to detect a face inside
initImg = cv2.imread('RedCarpetFace1.jpg')

#*** for opencv to use image correctly mustfirst convert to grayscale
gsImg = cv2.cvtColor(initImg, cv2.COLOR_BGR2GRAY) #<---- second parameter sets img background to gray

#*** Detect Facial features within image
faceCoordinates = preTrainedFaceData.detectMultiScale(gsImg) #<--- the classifier trained the "preTrainedFaceData" variable to detect faces accuratley due to funneling it multiple positive and negative data sets (imgs with faces and without)
                                    #^^^ multiscale method allows images to be various dimensions (scales)
print(faceCoordinates)#<--- prints coordinates of facial borders to console (terminal)


#*** Draw Rectangle around detected face
cv2.rectangle(initImg, (68,178), (68+334,178+334), (0, 255, 0), 3)  #<--- (img, (x, y), (x+w, y+w), (B, G, R), BorderWidth)



#*** display in image view grayscaled image
cv2.imshow('AI Face detector', initImg) #<--- parameter 1 is AppBar title and 2 is Image to Open
cv2.waitKey() #<--- waitKey to prevent photo viewer from instantly exiting