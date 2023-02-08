import cv2 as cv                                                     # Module import name for opencv-python
import numpy as np
img=cv.imread('/home/yantravision/Desktop/opencv/opencv sample.jpg') # Read the image with location 
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)                              # convert our original image from the BGR color space to gray
cv.imshow('gray',gray)                                               # Displaying the image
# algorithm that can detect objects in images, irrespective of their scale in image and location
har=cv.CascadeClassifier('/home/yantravision/Desktop/opencv/harcascade.xml')
# Detects objects of different sizes in the input image.
fact_react=har.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
print(f'Num of face found = {len(fact_react)}')
cv.waitKey(0)                                                        # Delay for the iamge to display "0" represents inifine time.
