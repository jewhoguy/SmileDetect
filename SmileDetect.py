# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 15:23:19 2018

@author: Juho Laukkanen
"""
import numpy as np
import os
import cv2
import glob
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt

def readImages(path):     
    images = [] #pre-allocation   
    for filename in glob.glob(os.path.join(path,'*.jpg')): #read as: "for each filename in filepath"
        im = cv2.imread(filename) #read image
        resizedImages = cv2.resize(im,(64,64)) #resize images to smaller
        images.append(resizedImages) #save into variable
    return np.array(images,dtype='float32') #return array
    
def readLabels(path):
    data = open(path,"r") #open and read-only
    labels = [] #pre-allocation
    yaw = []
    pitch = []
    roll = []
    for line in data:
        lineData = line.split(' ') #split text in file
        labels.append(lineData[0]) #label data in first column (1's and 0's)
        yaw.append(lineData[1]) #yaw
        pitch.append(lineData[2]) #pitch
        roll.append(lineData[3]) #roll
#        data.close()
    return labels

#imageFolder = "E:\\Smile Detection\\files"
images = readImages("F:\\Smile Detection\\files")
#labelFolder = "E:\\Smile Detection\\labels.txt"
labels = readLabels("F:\\Smile Detection\\labels.txt")

#data format manipulation, NN's work better with values between 0's and 1's
images = np.array(images,dtype='float32')/255
labels = np.array(labels,dtype='int32')

#classes to vector
nb_classes = 2 #number of classes
labels = np_utils.to_categorical(labels,nb_classes).astype(np.float32)
input_shape = images[1].shape

#shuffle data
indices = np.arange(len(images))
np.random.shuffle(indices)
images = images[indices]
labels = labels[indices]

#Weights
class_totals = labels.sum(axis=0)
class_weight = class_totals.max()/class_totals

print('Input datatype:',images.dtype,'min:',images.min(),'max:',images.max(),'shape:',images.shape)
print('Label data type:',labels.dtype,'min:',labels.min(),'max:',labels.max(),'shape:',labels.shape)
#Define CNN model
model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=input_shape,activation='relu',padding='same'))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(Dropout(0.25))

model.add(Conv2D(32,(3,3),padding='same',activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(Dropout(0.25))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(nb_classes,activation='softmax'))
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

epochs = 100
validation_split = 0.2
X_train, x_test, Y_train, y_test = train_test_split(images,labels,test_size = validation_split,random_state=42)

model.fit(X_train,Y_train,batch_size = 128,class_weight=class_weight,verbose=1, epochs=epochs,validation_data = (x_test,y_test))#validation_split=validation_split)

#model.save("SmileDetect_complex.h5py")
open('model.json', 'w').write(model.to_json())
model.save_weights('weights.h5')

plt.plot(model.model.history.history['loss'])
plt.plot(model.model.history.history['acc'])
plt.plot(model.model.history.history['val_loss'])
plt.plot(model.model.history.history['val_acc'])
plt.legend(['loss','accuracy','validation loss','validation accuracy'])
plt.show()


n_validation = int(len(X_train) * validation_split)
y_predicted = model.predict(X_train[-n_validation:])
print('AUC score:',roc_auc_score(Y_train[-n_validation:], y_predicted))

test_eval = model.evaluate(x_test,y_test,verbose = 0)
print('Test accuracy:', test_eval[1]*100)
