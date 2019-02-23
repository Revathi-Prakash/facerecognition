import cv2
import sqlite3
import numpy as np

faceDetect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
cam=cv2.VideoCapture(0)

def insertOrUpdate(id,name):
     conn = sqlite3.connect("FaceBase.db")
     cmd =  "SELECT * FROM Users where ID="+str(id)
     #cmd="UPDATE people SET Name=' "+str(name)+" ' WHERE ID="+str(id)
     cursor = conn.execute(cmd)
     isRecordExist = 0
     
     for row in cursor:
        isRecordExist = 1
     if(isRecordExist ==1):
         cmd="UPDATE Users SET Name=' "+str(name)+" ' WHERE ID="+str(id)
     else:
         cmd="INSERT INTO Users(ID,Name) Values("+str(id)+",' "+str(name)+" ' )"
     conn.execute(cmd)
     conn.commit()
     conn.close()    

        
    
id = input("enter user id")
name = input("Enter user name")
'''
branch = input("enter branch")
year = input("enter the year")
entry = input("whether the user is allowed to enter")
'''
insertOrUpdate(id,name)
sampleNum = 0;
while(True):
    ret,img = cam.read();
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        sampleNum +=1;
        cv2.imwrite("dataSet/User." + str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.waitKey(100);
    cv2.imshow("Face",img);
    cv2.waitKey(1);
    if(sampleNum > 30):
        break;
cam.release();
cv2.destroyAllWindows();  
