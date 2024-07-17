# 034803102018
# Here's lies a prototipe code for my first project on opencv and
# linux environment. it's bit rusty and klinky but, it's  yours.
# 
#
#
# Truly,
# SF 

import cv2
import numpy as np

# define ROI (region of interest)
w , dw, h, dh =  315, 20, 115, 20
track_window = (w, dw, h, dh)

# define the color window
COLOR_ROWS = 60
COLOR_COLS = 250

# command to get frame/video from camera attached to raspeberry pi
cap = cv2.VideoCapture(0)
if not cap.isOpened():
	raise RuntimeError('Error opening VideoCapture')

# set video stream/frame width by 640 and height by 480 
cap.set(3, 640)
cap.set(4, 480)

# define colorArray
colorArray = np.zeros((COLOR_ROWS, COLOR_COLS, 3), dtype=np.uint8)
cv2.imshow('Color', colorArray)

# Create windows for display
cv2.namedWindow('Color', cv2.WINDOW_NORMAL)
cv2.namedWindow('ViOut', cv2.WINDOW_NORMAL)
#cv2.namedWindow('Result', cv2.WINDOW_NORMAL)

# Moved the window to a spesific
cv2.moveWindow('Color', 40, 270) 
cv2.moveWindow('ViOut', 0, 0)
#cv2.moveWindow('Result', 680, 0)

while (True) :
	ret, frame = cap.read ()
	if not ret :
		break
	# if frame acquired correctly, ret will assign by true
	
	# show the color window
        colorArray[:] = frame[240, 320, :]
        rgb = frame[240, 320, [2,1,0]]
        
        # From stackoverflow.com/questions/1855884/determine-font-color-based-on-background-color
        luminance = 1 - (0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]) / 255
        if luminance < 0.5:
            textColor = [0,0,0]
        else:
            textColor = [255,255,255]

        cv2.putText(colorArray, str(rgb), (20, COLOR_ROWS - 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=textColor)
	#cv2.moveWindow('Color', 155, 90)
	cv2.imshow('Color', colorArray)

	# convert bgr color frame to hsv format
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# Create Resizable window with smallest resolution 320x240
	cv2.resizeWindow('ViOut', 320, 240)
	#cv2.resizeWindow('Result', 320, 240)

	# display the frame acquired 
	cv2.imshow('ViOut' , frame)
	#cv2.imshow('Result', res)


	# wait for key to be pressed for 1 ms
	k = cv2.waitKey(1) & 0xFF
	if k == 27: # ASCII code for ESC button on keyboard
		break # break 'while' loop

# if everything is done, then close all windows and release camera
cv2.destroyAllWindows()
cap.release()



