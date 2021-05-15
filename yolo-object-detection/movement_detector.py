import numpy as np
import argparse
import time
import cv2
import os
from imutils.object_detection import non_max_suppression
import pytesseract
from matplotlib import pyplot as plt
import telegram_send

i = 0

input = cv2.VideoCapture('videos/test01.mp4')

ret, capture1 = input.read()
ret, capture2 = input.read()

                # Setup for COCO car detection # 

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", default='yolo-coco',
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

                # End for COCO setup # 

while input.isOpened():

    diff = cv2.absdiff(capture1, capture2)                                                    #The apsolute difference between each of the captures.
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)                                        #Transfering the differences to grayscale. 
    diff_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)                                        #Applying Gaussian blur to the grayscale. 
    _, thresh = cv2.threshold(diff_blur, 20, 255, cv2.THRESH_BINARY)                          #Extracting treshold. 
    diff_dilated = cv2.dilate(thresh, None, iterations=3)                                     #Dialating the threshold. 
    contours, _ = cv2.findContours(diff_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)      #Applying contours on that dilation. 

    for contour in contours:                                                                  #Finding and applying the boundariy boxes. 
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 12000:
            continue

        #cv2.rectangle(capture1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cropped = capture1[y:y+h, x:x+w]
        #cv2.imwrite('cropped.png', cropped)
        #image = cv2.imread('images/here.jpg', 0)

        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        
        blob = cv2.dnn.blobFromImage(cropped, 1 / 255.0, (416, 416),
	    swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
<<<<<<< HEAD
                print("0")
                if confidence > args["confidence"]:
                    print("!found a vroom vroom!")
                    telegram_send.send(messages=["Car detected"])
=======
                print("found nothing")
                if confidence > args["confidence"]:
                    print("!found a vroom vroom!")
                    telegram_send.send(messages=["Wow that was easy!"])
>>>>>>> 1e15ad9308618dcf760c44263dc607306efafab3
                    
                    break

    #cv2.imshow("output", capture1)

    #img = cv2.imread('cropped.png', 0)
    #cv2.imshow('cropped', img)

    capture1 = capture2

    ret, capture2 = input.read()                                                             # THIS IS THE END OF THE MOVEMENT CAPTURE AND IMG EXTRACTION !
    
    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
input.release()
