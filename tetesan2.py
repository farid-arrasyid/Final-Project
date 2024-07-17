
#Cek fungsionalitas sensor tetesan di RasPi
#3-2-2019 - Talitha Yusli Alifia


#import numpy as np
import RPi.GPIO as GPIO
import time


GPIO.setmode(GPIO.BCM)
i=0


#def my_callback(channel):
    #i=i+1
    #print('Jumlah Tetesan : {0}'.format(i))

GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_UP)
#GPIO.add_event_detect(18, GPIO.FALLING, callback=my_callback)
temp = 0

while True:
    if temp == 1 and GPIO.input(18) == 0 :
        i = i+1
        print('Jumlah Tetesan : {0}'.format(i))
    temp = GPIO.input(18)
    
    #if KeyboardInterrupt() :
       # break
 # if k == ord('q') :
 
