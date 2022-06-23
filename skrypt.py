from cv2 import ORB_create
from rozw import solv
from tab import obr
import gcode_sender 
import cv2
import time
import serial
import gcode_sender

img = cv2.imread(r'C:\Users\pauli\OneDrive\Pulpit\pytoon\Nowy folder\sudoku.png')
tab1 = obr(img)
tab2 = obr(img)
solv(tab1,tab2)
tab = tab2 - tab1
print (tab)
gcode_sender.openn()
gcode_sender.send_line("G90")
gcode_sender.send_line("G28")
# send_line("G21")
gcode_sender.send_line("G00 Z5.000000")
gcode_sender.send_line("G90")
sss=[]
sss.append("")
sss.append(r"\1.gcode")
sss.append(r"\2.gcode")
sss.append(r"\3.gcode")
sss.append(r"\4.gcode")
sss.append(r"\5.gcode")
sss.append(r"\6.gcode")
sss.append(r"\7.gcode")
sss.append(r"\8.gcode")
sss.append(r"\9.gcode")
odleglosc = 22.5



for x in range(9):
    for y in range(9):
        if tab[x][y]!=0:
            sciezka=r"C:\Users\pauli\OneDrive\Pulpit\ploter\cyfry"+sss[tab[x][y]]
            gcode_sender.send_line("G00 X"+str(y*odleglosc)+" Y"+str(((8-x)*odleglosc))+"F5000")
            gcode_sender.send(sciezka)
        # gcode_sender.send_line("G00 X"+str(y*odleglosc)+" Y"+str(((8-x)*odleglosc)))      

gcode_sender.send_line("G90")
gcode_sender.send_line("G00 X0 Y0 Z5")


# zamknij port
gcode_sender.close
print("czas poświęcony obliczeniom", time.process_time())