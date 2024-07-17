#Test with ROI positioned first
#6-2-2019 Talitha Yusli Alifia

import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
i=0


V=0.05

dir='Test\\'
COLOR_ROWS = 60
COLOR_COLS = 250
cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError('Error opening VideoCapture')

cap.set(3, 640)
cap.set(4, 480)

cv2.namedWindow('ViOut', cv2.WINDOW_NORMAL)
cv2.moveWindow('ViOut', 50, 50)

cv2.namedWindow('Color', cv2.WINDOW_NORMAL)
cv2.moveWindow('Color', 690, 50)

cv2.moveWindow('Result', 700, 300)


x=215
y=295

w=50
h=50

test_color=np.zeros((1,1,3), dtype=np.uint8)
hsv=test_color
colorArray = np.zeros((COLOR_ROWS, COLOR_COLS, 3), dtype=np.uint8)
n_colors = 5
def imaging():
    global hsv
    global test_color
    global colorArray
    global test
    y1=y+w
    x1=x+h
    ret, frame=cap.read()
    test = frame[x-2:x1+2, y-2:y1+2, :]
    cv2.imshow('Cropped Image', test)
    
    if not ret:
        raise
    frame=cv2.rectangle(frame,(y1+2, x1+2), (y-2, x-2), (255, 0, 0), 2)
    cv2.imshow('ViOut', frame)

    pixels = np.float32(test.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2. TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    test_color[0,0,:] = dominant
    hsv = cv2.cvtColor(test_color, cv2.COLOR_BGR2HSV)
    rgb = test_color[0,0,[2, 1, 0]]  
    colorArray[:,:]=test_color[0,0]
    luminance = 1-(0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2])/255
    textColor = [0, 0, 0] if luminance<5 else [255, 255, 255]
    cv2.putText(colorArray, str(hsv[0,0]), (20, COLOR_ROWS-20),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=textColor)
    cv2.imshow('Color', colorArray)

def positioning():
    global x
    global y
    while (True):
        k=cv2.waitKey(1)&0xFF
        if k==ord('q'):
            return
        elif k==ord('s'):
            x+=5
        elif k==ord('w'):
            x-=5
        elif k==ord('a'):
            y+=5
        elif k==ord('d'):
            y+=5
        imaging()    

n=0
def my_callback():
    global n
    global test
    n+=1
    result=np.zeros((COLOR_ROWS, COLOR_COLS+50, 3), dtype=np.uint8)
    cv2.imwrite(dir+"{0}.JPG".format(n),test[2:52, 2:52,:])
    cv2.putText(result, 'Drop: {0} V: {1} mL'.format(n,n*V), (20, COLOR_ROWS-20),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=[255, 255, 255])
    cv2.imshow('Result', result)
    file.write("{0}\t{1}\t{2}\t{3}\n".format(hsv[0,0,0], hsv[0,0,1], hsv[0,0,2], n))
    

positioning()
file=open(dir+"test.txt","w")
GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.add_event_detect(18, GPIO.FALLING, callback=my_callback)

while (True):
    imaging()
    k=cv2.waitKey(1)&0xFF
    if k==ord('p'):
        positioning()
    elif k==27:
        break
#    elif k==ord('i'):
#        my_callback()
    if i>=50:
        GPIO.cleanup(18)
        break
file.close()
cv2.destroyAllWindows()
cap.release()