import numpy as np
import cv2

# Capture the frames of the video
cap = cv2.VideoCapture('input.mov')

# Cars classifiere
car_cascade = cv2.CascadeClassifier('cars.xml')

# As long as thee capture is still open  then the loop will run
while(cap.isOpened()):
	# Read the frames of the video
    ret, frame = cap.read()

     # Convert each frame to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Show the video in a frame
    cv2.imshow('frame',frame)

    # Break out of the loop on quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# De-allocate any associated memory usage
cap.release()
cv2.destroyAllWindows()