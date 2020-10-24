# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 18:53:46 2020

@author: DiyaM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
from mtcnn.mtcnn import MTCNN
import cv2
from tensorflow.keras.models import load_model

#Global variables and paths
path=os.getcwd()+"/"

#Reading data from face.npz file saved from face_detection.py
data = np.load(path+"faces.npz", allow_pickle=True)
X=data['arr_0']

y_classes=data['arr_1']

#plt.imshow(X[4])
#print(X[4])
#getting the categorical int from y_classes
y=pd.Series(y_classes, dtype='category').cat.codes.values
#print(y_classes[4])

    
model = load_model('facenet_keras.h5')

# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = np.expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

newTrainX = list()
i = 0
for face_pixels in X:
    #print(face_pixels)
    #print(i)
    #i=i+1
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)
newTrainX = np.asarray(newTrainX)
print(newTrainX.shape)


np.savez_compressed('embedding_face.npz', newTrainX, y)
print("embedding created")

    
    

