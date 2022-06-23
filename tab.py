import cv2
import pytesseract
import numpy as np
from numpy import*
from keras.models import load_model


# from sklearn.model_selection import train_test_split
# from keras.preprocessing.image import ImageDataGenerator
# from keras.utils.np_utils import to_categorical

# from keras.models import Sequential
# from keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
# from keras.layers import Dropout, Flatten
# from keras.layers.convolutional import Conv2D, MaxPooling2D
model1 = load_model('my_model.h5')



def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

def obr(img):


    pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\tesseract.exe"
    # img = cv2.imread(r'C:\\Users\\raven\\OneDrive\\Pulpit\\pythony\\sudoku.png')
    img = cv2.resize(img, (900, 900))
    # img = img[9:790, 9:790]
    # cv2.imshow('image2', img) 
    # Exiting the window if 'q' is pressed on the keyboard.
    if cv2.waitKey(0) & 0xFF == ord('q'): 
        cv2.destroyAllWindows()
    img = cv2.resize(img, (900, 900))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # kernel = np.ones((1, 1), np.uint8)
    # img = cv2.dilate(img, kernel, iterations=1)
    # img = cv2.erode(img, kernel, iterations=1)
    # img = cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    tab=zeros((9,9), int)
    for i in range(9):
        for j in range(9):
            x1=int(((900/9)*i))+10
            x2=int((900/9)*(i+1))-10
            y1=int(((900/9)*j))+10
            y2=int((900/9)*(j+1))-10
            crop_img = img[x1:x2, y1:y2]
            
            img5 = np.asarray(crop_img)
            img5 = cv2.resize(img5,(32,32))
            img5 = preProcessing(img5)
            img5 =  img5.reshape(1,32,32,1)
            predict_x=model1.predict(img5) 
            classes_x=np.argmax(predict_x,axis=1)
            if np.amax(predict_x)>0.8:
                tab[i][j] = int(classes_x)
            else:
                tab[i][j] = 0
            
            # print (classes_x, np.amax(predict_x))
    #         cv2.imshow('image2', crop_img) 
    # # Exiting the window if 'q' is pressed on the keyboard.
    #         if cv2.waitKey(0) & 0xFF == ord('q'): 
    #             cv2.destroyAllWindows()
    print (tab)
    # img5 = np.asarray(crop_img)
    # img5 = cv2.resize(img5,(32,32))
    # img5 = preProcessing(img5)
    # img5 =  img5.reshape(1,32,32,1)
    # predict_x=model1.predict(img5) 
    # classes_x=np.argmax(predict_x,axis=1)
    # print (classes_x, np.amax(predict_x))
    return tab