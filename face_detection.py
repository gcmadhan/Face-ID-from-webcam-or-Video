import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import sys, os
from mtcnn.mtcnn import MTCNN
import cv2

#Global variables & MTCNN detector

#detector = MTCNN()

#result = detector.detect_faces(image)

detect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

m_path =os.getcwd()+"/"
path = m_path+"data_set/train/"
img_size = (200,200)
data_x = []
y=[]

#extract function to extract the face from image. 
def extract_face(file_name):
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detect.detectMultiScale(img)
    
    for (x,y,w,h) in faces:
        #x, y, w, h = faces
        out_image = img[y:y+h, x:x+w]
        out_image = cv2.resize(out_image, (160,160))
        print(file_name + "  completed ......" )
        #print(out_image)
        return out_image

#function to read the folder and extract the images from each folder and update the image array
def load_data(path):
    #print(path)
    for i in os.listdir(path):
        #print(i)
        for sub_dir in os.listdir(path+"/"+i):
            print("Reading images from: "+i+" file name : " + sub_dir)
            #print(sub_dir) 
            dir_path = path+i+'/'+sub_dir
            #print(dir_path)
            face= extract_face(dir_path)
            if (isinstance(face,type(None))):
                continue
            
            data_x.append(face)
            #data_x.append(extract_face(dir_path))
            y.append(i)
    print("Face detection completed .......")
    return np.asarray(data_x),np.asarray(y)
            

#extract face image is stored in image array and saved as faces.npz compressed format.     
print(path) 
X, y = load_data(path)
print(X.shape)
np.savez_compressed(m_path+"faces.npz",X,y)
print("Face_details updated.....")