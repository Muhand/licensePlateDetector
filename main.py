import numpy as np
import cv2

# Capture the frames of the video
cap = cv2.VideoCapture('input.mov')

# As long as thee capture is still open  then the loop will run
while(cap.isOpened()):
	# Read the frames of the video
    ret, frame = cap.read()

    # Show the video in a frame
    cv2.imshow('frame',frame)

    # Break out of the loop on quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# De-allocate any associated memory usage
cap.release()
cv2.destroyAllWindows()