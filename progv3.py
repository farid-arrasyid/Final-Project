

import cv2
import numpy as np


# define ROI (region of interest)
w , dw, h, dh =  315, 20, 115, 20
track_window = (w, dw, h, dh)

# define the color window
COLOR_ROWS = 60
COLOR_COLS = 250

# command eito get frame/video from camera attached to raspeberry pi
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
#cv2.moveWindow('Color', 40, 270) 
#cv2.moveWindow('ViOut', 0, 0)
#cv2.moveWindow('Result', 680, 0)
x = 159
h = 50

y = 89
w = 50




while (True) :
	y1 = y+w
	x1 = x+h
	ret, frame = cap.read ()
	if not ret :
		break
	
	# if frame acquired correctly, ret will assign by true
	test = frame[x:x1,y:y1, :]
	
	test_color= np.zeros((1, 1, 3), dtype=np.uint8)
	cv2.imshow('Cropped Image',test)
	frame= cv2.rectangle(frame,(y1, x1), (y, x), (255, 0, 0), 2)
	# show the color window
	#for i in range(3):
		
	#	test_color[0,0, i] = test[:,:,i].mean()
	#	test_color[0,0, i] = test_color[0,0,i].astype(int)
	
	#test_color[0,0,:]= test.mean(axis=0).mean(axis=0)
	pixels = np.float32(test.reshape(-1, 3))	
	n_colors = 5
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
	flags = cv2.KMEANS_RANDOM_CENTERS

	_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
	_, counts = np.unique(labels, return_counts=True)
	#test_color[0,0,:]= test.mean(axis=0).mean(axis=0)
	colorArray[:,:]= test_color[0,0]
	dominant = palette[np.argmax(counts)]
	test_color[0,0,:] =dominant
	colorArray[:,:]= test_color[0,0]
	hsv = cv2.cvtColor(test_color, cv2.COLOR_BGR2HSV)
	rgb = test_color[0,0,[2,1,0]]
        print str(rgb)
        
        # From stackoverflow.com/questions/1855884/determine-font-color-based-on-background-color
        luminance = 1 - (0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]) / 255
        if luminance < 0.5:
            textColor = [0,0,0]
        else:
            textColor = [255,255,255]

        cv2.putText(colorArray, str(hsv[0,0]), (20, COLOR_ROWS - 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=textColor)
	
	#cv2.moveWindow('Color', 155, 90)
	cv2.imshow('Color', colorArray)
	# convert bgr color frame to hsv format
	#hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# Create Resizable window with smallest resolution 320x240
	#cv2.resizeWindow('ViOut', 320, 240)
	#cv2.resizeWindow('Result', 320, 240)

	# display the frame acquired 
	cv2.imshow('ViOut' , frame)
	#cv2.imshow('Result', res)


	# wait for key to be pressed for 1 ms
	k = cv2.waitKey(1) & 0xFF
	if k == 27: # ASCII code for ESC button on keyboard
		break # break 'while' loop
	elif k == ord('s'):
		x += 5
	elif k == ord('w'):
		x -= 5
	elif k == ord('a'):
		y -= 5
	elif k == ord('d'):
		y += 5
		

# if everything is done, then close all windows and release camera
cv2.destroyAllWindows()
cap.release()



