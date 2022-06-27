from cv2 import ORB_create
from solver import solv
from tab import obr
import cv2
from numpy import*
import numpy as np
from keras.models import load_model
import os

####################### USTAWIENIA
uczenie = 0

def preProcess(img):
    # przekonwertowanie obrazu na skalę szarości
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    # dodanie szumu Gaussa
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  
    # włączenie adaptacyjnego progu detekcji
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)  
    return imgThreshold

### zmiana perspektywy
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

### Wykrycie największego konturu
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area

### podział wykrytego obrazu sudoku na pojedyńcze komórki
def splitBoxes(img):
    rows = np.vsplit(img,9)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes

### detekcja znaku
def obr(img):
    x=0
    img5 = np.asarray(img)
    img5 = cv2.resize(img5,(32,32))
    img5 = cv2.equalizeHist(img5)/255
    img5 =  img5.reshape(1,32,32,1)
    predict_x=model1.predict(img5) 
    classes_x=np.argmax(predict_x,axis=1)
    if uczenie == 1:
        if np.amax(predict_x)>0.7:
            print( int(classes_x))
            while os.path.isfile("C:/Users/raven/OneDrive/
            Pulpit/pythony/cyfry"+"/"+str(int(classes_x))+"/"+str(x)+".png"):
                x=x+1
            cv2.imwrite("C:/Users/raven/OneDrive/Pulpit/pythony/cyfry"+
            "/"+str(int(classes_x))+"/"+str(x)+".png", img)
        else:
            print(0)
            print(np.amax(predict_x))
            while os.path.isfile("C:/Users/raven/OneDrive/Pulpit/pythony
            /cyfry"+"/"+'0'+"/"+str(x)+".png"):
                x=x+1
            cv2.imwrite("C:/Users/raven/OneDrive/Pulpit/pythony/cyfry"+"/"+
            '0'+"/"+str(x)+".png", img)
    return classes_x