import numpy as np
import argparse
import time
import cv2
import os
from imutils.object_detection import non_max_suppression
import pytesseract
from matplotlib import pyplot as plt
import requests

txt0 = 'https://api.telegram.org/bot1794107664:AAH8IZuCdcLDr_koMY24otS3K8WbAHs7Ljw/sendMessage?chat_id=-431037595&text="Runnning camera..."'
txt1 = 'https://api.telegram.org/bot1794107664:AAH8IZuCdcLDr_koMY24otS3K8WbAHs7Ljw/sendMessage?chat_id=-431037595&text="a vehicle has been detected:"'
txt2 = 'https://api.telegram.org/bot1794107664:AAH8IZuCdcLDr_koMY24otS3K8WbAHs7Ljw/sendMessage?chat_id=-431037595&text="The number plate is: HEYYYYYA"'
#files={'photo':open('yolo-object-detection\images\mini.jpeg', 'rb')}

i = 0

input = cv2.VideoCapture('videos\stock1.mp4')

ret, capture1 = input.read()
ret, capture2 = input.read()

                # Setup COCO car detection # 

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", default='yolo-coco', 
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.3,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
#labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
labelsPath = 'yolo-coco\coco.names'
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

                  # End COCO setup #                   

            # Characther recognition setup #
             
# Returns a bounding box and probability score if it is more than minimum confidence
def predictions(prob_score, geo):
	(numR, numC) = prob_score.shape[2:4]
	boxes = []
	confidence_val = []

	# loop over rows
	for y in range(0, numR):
		scoresData = prob_score[0, 0, y]
		x0 = geo[0, 0, y]
		x1 = geo[0, 1, y]
		x2 = geo[0, 2, y]
		x3 = geo[0, 3, y]
		anglesData = geo[0, 4, y]

		# loop over the number of columns
		for i in range(0, numC):
			if scoresData[i] < args["min_confidence"]:
				continue

			(offX, offY) = (i * 4.0, y * 4.0)

			# extracting the rotation angle for the prediction and computing the sine and cosine
			angle = anglesData[i]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# using the geo volume to get the dimensions of the bounding box
			h = x0[i] + x2[i]
			w = x1[i] + x3[i]

			# compute start and end for the text pred bbox
			endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
			endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
			startX = int(endX - w)
			startY = int(endY - h)

			boxes.append((startX, startY, endX, endY))
			confidence_val.append(scoresData[i])

	# return bounding boxes and associated confidence_val
	return (boxes, confidence_val)

          # End characther recognition setup # 

                    # !START OF MAIN! #

#requests.get(txt0)                                                                            #Text to show that we have initialized the main program. 
while input.isOpened():

    diff = cv2.absdiff(capture1, capture2)                                                    #The apsolute difference between each of the captures.
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)                                        #Transfering the differences to grayscale. 
    diff_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)                                        #Applying Gaussian blur to the grayscale. 
    _, thresh = cv2.threshold(diff_blur, 20, 255, cv2.THRESH_BINARY)                          #Extracting treshold. 
    diff_dilated = cv2.dilate(thresh, None, iterations=3)                                     #Dialating the threshold. 
    contours, _ = cv2.findContours(diff_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)      #Applying contours on that dilation. 

    for contour in contours:                                                                  #Finding and applying the boundariy boxes. 
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 1000:
            continue

        cv2.rectangle(capture1, (x, y), (x+w, y+h), (0, 255, 0), 2)
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
                print("null")
                if confidence > args["confidence"]:

                    #Creating argument dictionary for the default arguments needed in the code. 
                    args = {"image":"heyy.jpg", "east":"frozen_east_text_detection.pb", "min_confidence":0.5, "width":320, "height":320}

                    #Give location of the image to be read.

                    args['image']="images\heyy.jpg"
                    image = cv2.imread(args['image'])

                    #Saving a original image and shape
                    orig = image.copy()
                    (origH, origW) = image.shape[:2]

                    # set the new height and width to default 320 by using args #dictionary.  
                    (newW, newH) = (args["width"], args["height"])

                    #Calculate the ratio between original and new image for both height and weight. 
                    #This ratio will be used to translate bounding box location on the original image. 
                    rW = origW / float(newW)
                    rH = origH / float(newH)

                    # resize the original image to new dimensions
                    image = cv2.resize(image, (newW, newH))
                    (H, W) = image.shape[:2]

                    # construct a blob from the image to forward pass it to EAST model
                    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)

                    # load the pre-trained EAST model for text detection 
                    net = cv2.dnn.readNet(args["east"])

                    # The following two layer need to pulled from EAST model for achieving this. 
                    layerNames = [
                        "feature_fusion/Conv_7/Sigmoid",
                        "feature_fusion/concat_3"]
                    
                    #Forward pass the blob from the image to get the desired output layers
                    net.setInput(blob)
                    (scores, geometry) = net.forward(layerNames)

                    # Find predictions and  apply non-maxima suppression
                    (boxes, confidence_val) = predictions(scores, geometry)
                    boxes = non_max_suppression(np.array(boxes), probs=confidence_val)

                    ##Text Detection and Recognition 

                    # initialize the list of results
                    results = []

                    # loop over the bounding boxes to find the coordinate of bounding boxes
                    for (startX, startY, endX, endY) in boxes:
                        # scale the coordinates based on the respective ratios in order to reflect bounding box on the original image
                        startX = int(startX * rW)
                        startY = int(startY * rH)
                        endX = int(endX * rW)
                        endY = int(endY * rH)

                        #extract the region of interest
                        r = orig[startY:endY, startX:endX]

                        #configuration setting to convert image to string.  
                        configuration = ("-l eng --oem 1 --psm 8")
                        ##This will recognize the text from the image of bounding box
                        text = pytesseract.image_to_string(r, config=configuration)

                        # append bbox coordinate and associated text to the list of results 
                        results.append(((startX, startY, endX, endY), text))

                    #Display the image with bounding box and recognized text
                    orig_image = orig.copy()

                    # Moving over the results and display on the image
                    for ((start_X, start_Y, end_X, end_Y), text) in results:
                        # display the text detected by Tesseract
                        print("{}\n".format(text))

                        # Displaying text
                        text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
                        cv2.rectangle(orig_image, (start_X, start_Y), (end_X, end_Y),
                            (0, 0, 255), 2)
                        cv2.putText(orig_image, text, (start_X, start_Y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0, 255), 2)

                    requests.post('https://api.telegram.org/bot1794107664:AAH8IZuCdcLDr_koMY24otS3K8WbAHs7Ljw/sendPhoto?chat_id=-431037595', files=orig_image)
                    break

    cv2.imshow("output", capture1)

    #img = cv2.imread('cropped.png', 0)
    #cv2.imshow('cropped', img)

    capture1 = capture2

    ret, capture2 = input.read()                                                             # THIS IS THE END OF THE MOVEMENT CAPTURE AND IMG EXTRACTION !
    
    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
input.release()
