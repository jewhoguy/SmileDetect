"""
@author: Juho Laukkanen
"""
import numpy as np
import cv2
from keras.models import model_from_json

def detect(frame,cascade,str):
    if str == 'face':
        detect = cascade.detectMultiScale(frame,scaleFactor=1.4,minNeighbors=3,minSize=(50,50),flags=cv2.CASCADE_SCALE_IMAGE)
        return detect
    elif str == 'mouth':
        detect = cascade.detectMultiScale(frame,scaleFactor=1.4,minNeighbors=15,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        return detect

#Number of faces found
    print('Faces found: ',len(detect))

def drawRects(frame,face_rects,smile_rects):
    for x,y,w,h in face_rects:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'Face',(x,y-7),3,1.2,(255,0,0),2,cv2.LINE_AA)
        for x1,y1,x2,y2 in smile_rects:
            cv2.rectangle(frame,(x1,y1),(x1+x2,y1+y2),(0,0,255),2)            
            cv2.putText(frame,'Mouth',(x1,y1-7),3,1.2,(255,0,0),2,cv2.LINE_AA)

def loadModel():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("weights.h5")
    print("Loaded model from disk")
    return model

#Load DCNN
model = loadModel()

#Create haar cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
#smile_cascade = cv2.CascadeClassifier('haarcascade_mouth.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

#Start video capture
#cap = cv2.VideoCapture('VID_1.mp4')
cap = cv2.VideoCapture('Roller Coaster.mp4')
#cap = cv2.VideoCapture('IMG_20181214_215656.jpg')
#cap = cv2.VideoCapture('file0011.jpg')
#Check if file opened
if (cap.isOpened()==False):
    print('Error opening video')

#While frames to process
while cap.isOpened():
    ret,img = cap.read()
#    img = cv2.resize(img,(1240,1020)) #resize to smaller resolution
    img = cv2.resize(img,(840,640))  
#    img = cv2.resize(img,(256,256))
#    img = np.rot90(img,1,axes=(1,0)) #rotate video frames
    frame = img.copy()    
    grey_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    if ret == True:
        face_rects = detect(grey_img,face_cascade,'face')
        smile_rects = detect(grey_img,smile_cascade,'mouth')
        print(smile_rects)
        drawRects(frame,face_rects,smile_rects)

    pred = model.predict(np.reshape(cv2.resize(frame,(64,64)),(1,64,64,3)))
    if pred[0][1] > pred[0][0]:
        print("Smile detected")
    else:
        print("No smile detected")
        
    cv2.imshow('frame',frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
#    else:
#        break

cap.release()
cv2.destroyAllWindows()

#        smile_rects = detect(grey_img,nested_cascade
#        drawRects(frame,smile_rects,(0,0,255))

#        if isinstance(face_rects,tuple):
#            continue
#        elif face_rects.shape[0] == 1:
#            smile_rects = smile_rects[1:]