from cv2 import ORB_create
from rozw import solv
from tab import obr
import gcode_sender 
import cv2
import time
import serial
import gcode_sender
import numpy as np
import cv2
import imutils
import requests
import numpy as np
import cv2
from numpy import*
import requests
import imutils
import pytesseract
import numpy as np
from numpy import*
from keras.models import load_model
import os
model1 = load_model('my_model1.h5')
gcode_sender.openn()
gcode_sender.send_line("G90")
gcode_sender.send_line("G28")
x=gcode_sender.baza()
# send_line("G21")
# gcode_sender.send_line("G00 Z5.000000")
gcode_sender.send_line("G90")
y=gcode_sender.baza()
while y==x:
    y=gcode_sender.baza()
    print(y)
    time.sleep(1)
gcode_sender.send_line("G00 Z5 F5000")
gcode_sender.send_line("G00 X0 Y250 Z5 F5000")
gcode_sender.send_line("G00 X250 Y0 Z5")
gcode_sender.send_line("G00 X0 Y0 Z5")
gcode_sender.send_line("G00 X250 Y100 Z5")