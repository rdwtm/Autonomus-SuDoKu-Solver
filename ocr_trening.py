import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import load_model
import matplotlib.pyplot as plt

##################
path = 'myData1'
testRatio = 0.2
valRatio = 0.2
imageDimensions = (32,32,3)

batchSizeVal=10
epochsVal= 20
stepsPerEpoch = 200

##################
images = []
classNo = []

myList = os.listdir(path)
print("total No of Classes Detected", len(myList))
print("Importing Classes.....")
noOFClasses = len(myList)

for x in range (0,noOFClasses):
    myPicList = os.listdir(path+"/"+str(x))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        if curImg is None:
            print('Wrong path:', path+"/"+str(x)+"/"+y)
        else:
            curImg = cv2.resize(curImg,(32,32))
            images.append(curImg)
            classNo.append(x)
    print(x,end= " ")
print(" ")


images = np.array(images)
classNo = np.array(classNo)

# print(images.shape)
# print(classNo.shape)

#### splittnig

X_train,X_test,y_train,y_test = train_test_split(images,classNo,
                                            test_size=testRatio)
X_train,X_validation,y_train,y_validation = train_test_split(X_train,
                                            y_train,test_size=valRatio)
print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)
numOfSamples = []
for x in range(0,noOFClasses):
    # print(len(np.where(y_train==x)[0]))
    numOfSamples.append(len(np.where(y_train==x)[0]))
print(numOfSamples)

def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

# img = preProcessing(X_train[30])
# img = cv2.resize(img,(300,300))
# cv2.imshow("preproc",img)
# cv2.waitKey(0)

X_train = np.array(list(map(preProcessing,X_train)))
X_test = np.array(list(map(preProcessing,X_test)))
X_validation = np.array(list(map(preProcessing,X_validation)))

print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],
                                                        X_validation.shape[2],1)

dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                                zoom_range=0.2, shear_range=0.1, 
                                rotation_range=10)
dataGen.fit(X_train)

y_train = to_categorical(y_train,noOFClasses)
y_test = to_categorical(y_test,noOFClasses)
y_validation = to_categorical(y_validation,noOFClasses)

def myModel():
    noOFFilters = 60
    sizeOfFilter1=(5,5)
    sizeOfFilter2= (3,3)
    sizeOfPool= (2,2)
    noOFnode = 500

    model = Sequential()
    model.add((Conv2D(noOFFilters,sizeOfFilter1,input_shape=(imageDimensions[0],
                                                            imageDimensions[1],1),
                                                            activation='relu')))

    model.add((Conv2D(noOFFilters,sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOFFilters/2,sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOFFilters/2,sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOFnode, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOFClasses,activation='softmax'))
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy', 
                                            metrics=['accuracy'])
    return model

model = myModel()
print(model.summary())

history = model.fit_generator(dataGen.flow(X_train,y_train,
                                    batch_size=batchSizeVal), 
                                    steps_per_epoch= stepsPerEpoch, 
                                    epochs= epochsVal,
                                    validation_data=(X_validation, y_validation),
                                    shuffle=1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['trening','walidacja'])
plt.title('zależność ilości epok od błędu uczenia')
plt.xlabel('ilość epok')
plt.ylabel('błąd uczenia')
plt.show()
print(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['trening','walidacja'])
plt.title('zależność ilości epok od precyzji modelu')
plt.xlabel('ilość epok')
plt.ylabel('precyzja')
plt.show()

score = model.evaluate(X_test,y_test,verbose=0)
print('test score', score[0] )
print('test accu', score[1])
model.save('my_model1.h5')