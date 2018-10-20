import numpy as np
import cv2
import imutils

# Capture the frames of the video
cap = cv2.VideoCapture('input.mov')

# Cars classifiere
car_cascade = cv2.CascadeClassifier('cars.xml')

# Initialize the first frame in the video stream
firstFrame = None
counter = 0
padding = 0

# As long as thee capture is still open  then the loop will run
while(True):
	# Read the frames of the video
    frame = cap.read()

    frame = frame[1]

    # if the frame could not be grabbed, then we have reached the end of the video
    if frame is None:
    	break

    # Resize the frame to 500, there is no need to resize a large frame
    frame = imutils.resize(frame, width=500)

    # Convert each frame to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply gaussian blur to smooth out the image
    gray = cv2.GaussianBlur(gray, (21,21), 0)

    # If the first frame is not initialized then initialize it
    if firstFrame is None:
    	firstFrame = gray
    	continue

    # compute the absolute difference between the current frame and first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image to fill in holes, then find contours on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # Loop over the contours
    for c in cnts:
    	# If the contour is too small then ignore it
    	if(cv2.contourArea(c) < 500):
    		continue

    	# Compute the boudning box for the contour, draw it on the frame
    	(x, y, w, h) = cv2.boundingRect(c)

    	# Create a new frame for this contour
    	blank_image = np.zeros((h,w,3), np.uint8)
    	extractedFrame = frame[y-padding:(y+h)+padding, x-padding:(x+w)+padding]
    	# if extractedFrame is not None:
    	# 	cv2.imshow('CARR',extractedFrame)
    	# 	continue
    	# else:
    	# 	break
    	name = 'CAR_' + str(counter)
    	cv2.imwrite('cars3/'+name+'.png',extractedFrame)

    	# Make sure it's a car by extracting all cars in this frame
    	cars = car_cascade.detectMultiScale(extractedFrame, 1.1, 1)
    	for (x,y,w,h) in cars:
    		# Draw a rectangle for each car
    		# cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2)

    		# Create the car frame
    		car = extractedFrame[y:y+h, x:x+w]

    		# Create a name for this car
    		name = 'CAR_' + str(counter)

    		# Write this image to a file
    		cv2.imwrite('cars2/'+name+'.png',car)
    	cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
    	counter+=1



    # Draw a rectangle around each car
    # for (x,y,w,h) in cars:
    #     cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    # Show the video in a frame
    cv2.imshow('Original',frame)
    cv2.imshow('Delta',frameDelta)
    cv2.imshow('Thresh',thresh)

    # Break out of the loop on quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# De-allocate any associated memory usage
cap.release()
cv2.destroyAllWindows()