import cv2
import os
import numpy as np
from PIL import Image
import pickle
import sqlite3



recognizer = cv2.face.LBPHFaceRecognizer_create()
cascadePath = "haarcascade_frontalface_default.xml";

faceCascade = cv2.CascadeClassifier(cascadePath);
recognizer.read('recognizer\\trainingData.yml')

path = 'dataSet'

def getProfile(id):
    conn = sqlite3.connect('FaceBase.db')
    cmd = "SELECT * FROM Users where ID=" + str(id)
    cur  = conn.cursor()
    cur.execute(cmd)
    cursor = cur.fetchall()
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
Id = 0
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        profile = getProfile(Id)
        if conf>50:
            if(profile!=None):
                cv2.putText(im,str(profile[0]), (x,y+h), font, 1, 255,2)
                cv2.putText(im,str(profile[1]), (x,y+h+30), font, 1, 255,2)
    cv2.imshow('im',im) 
    if cv2.waitKey(10) & 0xff==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
'''
rec = cv2.face.LBPHFaceRecognizer_create();
rec.read('recognizer\\trainingData.yml');
#faceDetect=cv2.CascadeClassifier("opencv/haarcascade_frontalface_default.xml");

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
path = 'dataSet'


def getProfile(id):
    conn = sqlite3.connect("FaceBase.db")
    cmd = "SELECT * FROM Users where ID=" + str(id)
    cursor = conn.execute(cmd)
    profile ="None"
    for row in cursor:
        profile = row
    conn.close()
    return profile
cam=cv2.VideoCapture(0);
font = cv2.FONT_HERSHEY_SIMPLEX
#font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
id = 0
while True:
    ret,img = cam.read();
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Face",gray);

    #faces = faceCascade.detectMultiScale(gray,1.3,5)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(100,100),flags = cv2.CASCADE_SCALE_IMAGE)
    #5);

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf = rec.predict(gray[y:y+h,x:x+w])
        print(conf)
        profile = getProfile(id)
        if(conf<50):
           
            if(profile=="None"):
                print(id)
            else:
                print("count")
        
            #cv2.cv.PutText(img,str(id),(x,y+h),font,255)
            #cv2.cv.PutText(cv2.cv.fromarray(img),str(profile[2]),(x,y+h+60),font,255)
        cv2.imshow("FACE",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
cam.release();
cv2.destroyAllWindows();  
 
 
'''
