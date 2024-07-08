from picamera import PiCamera
import numpy as np
from time import sleep

camera=PiCamera()
#camera.rotation=90
camera.resolution='2592x1944'

camera.start_preview()
sleep(180)
#for i in np.arange(0,60): 
#    camera.capture('/home/pi/Desktop/testimages/'+str(i)+'.jpg')
#    sleep(0.5)
camera.stop_preview()
#camera.capture('/home/pi/Desktop/distance.jpg')

print(camera.resolution)
camera.close()
