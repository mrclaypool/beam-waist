import time 
import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib
from picamera import PiCamera

GPIO.setmode(GPIO.BCM)

#LEDIO=26 #this is an LED for testing purposes
gatepower=16 #power for gate. must be set to one
gateIO=12 #signal from the gate (reads 0 if nothing detected)
camera=PiCamera() #tells the pi to use a camera

camera.resolution='2542x1944'

GPIOPins=[17,27,22,23] #pins for motor. IN1,IN2,IN3,IN4 on L298N
mymotor=RpiMotorLib.BYJMotor("MyMotorOne","Nema") #sets up motor. ("name","motor type")

#mymotor.motor_run(GPIOPins, 0.1, 5, False, False, "half", .05)
#args are (pins of motor, time to wait between steps, # steps, ccw, verbose, steptype, initdelay)

GPIO.setup(gateIO,GPIO.IN) #set up gate signal as an input
#GPIO.setup(LEDIO,GPIO.OUT) #set up the LED as an output (puts out voltage)

GPIO.setup(gatepower,GPIO.OUT) #sets up power for gate
GPIO.output(gatepower,1) #turns on gate power
#gatesig=GPIO.input(gateIO). this doesnt work. program checks it once
i=0
camera.start_preview()
try:
    while True:
        if  GPIO.input(gateIO)==0: #this must be called in loop
 #           GPIO.output(LEDIO,1)
            camera.capture('/home/pi/Desktop/7-3-24c/'+str(i)+'.jpg')
            mymotor.motor_run(GPIOPins,0.1,1,False,False,"half",.05)
            print ('Move, photo number'+str(i))
            time.sleep(0.5)
            
            i+=1
        else:
#            GPIO.output(LEDIO,0)
            print ('Stop. Breaking loop.')
            break
    time.sleep(0.1)

finally:
    GPIO.cleanup()
    camera.stop_preview()
    camera.close()

