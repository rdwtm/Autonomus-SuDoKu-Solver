from cv2 import ORB_create
from solver import solv
import gcode_sender 
import time
import cv2
from numpy import*
import requests
import numpy as np 
from keras.models import load_model
from narzedzia import*

##### USTAWIENIA
model1 = load_model('my_model1.h5')
url = "http://192.168.0.13:8080/shot.jpg" # adres serwera kamery
heightImg = 450
widthImg = 450
odleglosc = 17.3 # odległość między kolejnymi kratkami sudoku

#### Inicjacja komunikacji z ploterem
gcode_sender.openn("COM3")
gcode_sender.send_line("G90") # wsp. absolutne  
gcode_sender.send_line("G28") # bazowanie
# test zakończenia bazowania
x=gcode_sender.baza()
gcode_sender.send_line("G00 Z5.000000")
gcode_sender.send_line("G90")
y=gcode_sender.baza()
while y==x:
    y=gcode_sender.baza()
    print(y)
    time.sleep(1)

input("Press Enter to continue...")
gcode_sender.send_line("G00 Y100 Z1.000000") # 1 punkt ustawienia kartki
input("Press Enter to continue...")
gcode_sender.send_line("G00 Y255 Z1.000000") # 2 punkt ustawienia kartki
input("Press Enter to continue...")
gcode_sender.send_line("G00 Y0 Z5.000000 F5000")

### zapisanie zdjęcia z serwera kamery
img_resp = requests.get(url) 
img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
img = cv2.imdecode(img_arr, -1)

### przygotowanie obrazu
img = cv2.resize(img, (widthImg, heightImg))  # zmiana rozmiaru obrazu
imgThreshold = preProcess(img)                # binaryzacja obrazu

### wyszukanie konturów w obrazie
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, 
                                                cv2.CHAIN_APPROX_SIMPLE)

### Znalezienie największego konturu, czyli obrysu planszy sudoku
biggest, maxArea = biggestContour(contours)
tab1=zeros((9,9), int)
tab2=zeros((9,9), int)
if biggest.size != 0:
    biggest = reorder(biggest)
    # punkty do transformacji prespektywy
    pts1 = np.float32(biggest) 
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, 
                                                            heightImg]]) 
    matrix = cv2.getPerspectiveTransform(pts1, pts2) # GER
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

    ### podział obrazu na pojedyńcze komórki
    boxes = splitBoxes(imgWarpColored)

    for x in range(81):
        tab1[int(x/9)][x%9]=obr(boxes[x])
        tab2[int(x/9)][x%9]=obr(boxes[x])

solv(tab1,tab2)
tab = tab2 - tab1

# Wczytanie Gcodu cyfr
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

### Zapis cyfr na planszy
for x in range(9):
    for y in range(9):
        if tab[x][y]!=0:             # jeśli znaleziono znak(rozwiązanie) 
            sciezka=r"C:\Users\pauli\OneDrive\Pulpit\ploter\cyfry"+sss[tab[x][y]]
            gcode_sender.send_line("G00 X"+str(y*odleglosc)+
                    " Y"+str(((8-x)*odleglosc)+100)+"F5000")
            gcode_sender.send(sciezka)
        
### przezentacja rozwiązania
gcode_sender.send_line("G90")
gcode_sender.send_line("G00 X0 Y250 Z5")

### zamknij port
gcode_sender.close
print("czas poświęcony wykonaniu zadania", time.process_time())