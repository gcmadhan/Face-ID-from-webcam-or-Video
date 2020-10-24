# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 20:19:03 2020

@author: DiyaM
"""
import cv2
from tensorflow.keras.models import load_model
from sklearn.svm import SVC
import os
import numpy as np
import matplotlib.pyplot as plt

m_path=os.getcwd()+"/"
data = np.load(m_path+"embedding_face.npz")
X=data['arr_0']
y_classes=data['arr_1']
classes = os.listdir(m_path+"data_set/train/")
print("printing shape of X:", classes)

# fit model
model = SVC(kernel='linear', probability=True)
model.fit(X, y_classes)
model1 = load_model('facenet_keras.h5')



detect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

vid = cv2.VideoCapture(0) 
  
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, img = vid.read(0) 
    #gray = cv2.cvtColor(gray, code)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       
  
    faces = detect.detectMultiScale(img)
    
      
    
    print(faces)
    for (x,y,w,h) in faces:
        #cv2.crop
        print(x,y,w,h)
        img_pre = img[y:y+h, x:x+w]
        #cv2.imwrite("img1.jpeg",img_pre)
        img_pre = cv2.resize(img_pre, (160,160))
        cv2.imwrite("img1.jpeg",img_pre)
        img_pre = img_pre.astype('float32')
        #standardize pixel values across channels (global)
        mean, std = img_pre.mean(), img_pre.std()
        img_pre = (img_pre - mean) / std
        #img_pre = img_pre/160
        #print(img)
        img_pre = np.expand_dims(img_pre, axis=0)
        img_pre = model1.predict(img_pre)
        pre = model.predict(img_pre)
        prob = model.predict_proba(img_pre)
        print(prob.shape)
        
        #for i in range(len(pre)):
        name = classes[pre[0]]
        print_prob = max(prob[0])
        print("Name: ", name)
        print("Probability: ", print_prob)
        if (print_prob>0.60):
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            img = cv2.putText(img, name, (x-10,y-10),cv2.FONT_HERSHEY_PLAIN, color=(0,255,0),fontScale=2, thickness=2)
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = img[y:y+h, x:x+w]
    #    eyes = eye_cascade.detectMultiScale(roi_gray)
    
    # Display the resulting frame 
    cv2.imshow('frame', img) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 