#!/usr/bin/env python
# coding: utf-8


import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mtcnn.mtcnn import MTCNN
import sys,os
detector = MTCNN()


def extract_face(image):
    try:
        res = detector.detect_faces(image)
    except:
        return
    x, y, x1, y1 = res[0]['box']
    
    height = y+y1
    width = x+x1
    return x, y, height, width




def create_folder():
    name = input("Enter the name of the person:")
    path = os.getcwd()+"\data_set\\train"
    path = path+"\\"+name
    try:
        os.mkdir(path)
    except:
        print("Directory available")
    return name, path



def capture_faces():
    name, path =create_folder()
    cap = cv2.VideoCapture(0)
    res, frame = cap.read()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS,25)
    itr = 1
    while cap.isOpened():
        print(itr)
        x, y, height, width = extract_face(frame)
        #height = int(height +(height*0.15))
        #width = int(width + (width*0.05))
        #x=x+(x*0.30)
        #img_out = frame[y:height, x:width]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #img_out = cv2.resize(img_out, (200,200))
        plt.imsave(path+"\\"+"Madhan_"+str(itr)+".jpg", frame)
        img = cv2.rectangle(frame, (x,y),(width, height), (0,0,255), 2)

        img = cv2.putText(img, "Image #: "+str(x)+","+str(y+1),(50,50),cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0,0,255), 2 )
        if itr==15 :
            break
        cv2.imshow("windows", img)
        itr=itr+1

        res, frame =cap.read()
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 




capture_faces()






