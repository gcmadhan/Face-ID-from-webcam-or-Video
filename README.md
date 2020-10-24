# Identify face from Video/Webcam using Tensorflow & FaceNet
The idea of the project is to identify the faces in Video file or in webcam using Tensorflow FaceNet architecture & Skleanr Simple vector machine. Face recognition is being used in day to day applications like mobile authentication, attendance system, cctv, crowd monitoring etc. In this project, I have used haarcascades to extract the faces from datasets. Tensorflow Facent architecture is used to extract the features from the image, SVM is used to classify the image.

The project will create a Tensorflow model using the training dataset images categorised by folders. Model will predict the faces in the given images (test folder) and output will be placed in the output folder.

## Features
## Folder Structure

* Face Identification
  - data_set
      - train
        - Image category Folder 1
        - Image category folder 2
        - Image category folder 3
  - output
  - test


1.	Data_set/train – folder store training images segregated by folders. Ex: each category should be differentiated by folder
4.	Extract_face_frm.py – Python file executed to find the faces in Video/webcam.
5.	Create_embedding.py – python file is executed to create the embedding of the faces and store it in NPZ file.  
6.	create_face_database.py – python file is executed to create the database of the images. this will create folder for each person and store the images from Webcam. 
5.  face_detection.py - python file is used to extract the faces from the dataset and store it in image array as npz file. 
7.	Faces.npz – image array and classification array. Images are the faces extracted from the training folder images. 
8.  embedding_face.npz - npz file where the embedding of the images are stored and retrevied for identification purpose. 
9.	Requirment.txt – modules required to run the project
10.	Readme.md – readme file about the project. 

## Requirments
  - [x] Tensorflow 2.0
  - [x] Sklearn 
  - [x] Opencv
  - [x] Matplotlib
  - [x] sys, os
  - [x] download facenet pretrained model from internet. 

## Tensorflow
TensorFlow is a free and open-source software library for dataflow and differentiable programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networks.[4] It is used for both research and production at Google. TensorFlow was developed by the Google Brain team for internal Google use. It was released under the Apache License 2.0 on November 9, 2015
Refer:  https://www.tensorflow.org/api_docs/python/tf

## Convolutional Neural Network
A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.
The architecture of a ConvNet is analogous to that of the connectivity pattern of Neurons in the Human Brain and was inspired by the organization of the Visual Cortex. Individual neurons respond to stimuli only in a restricted region of the visual field known as the Receptive Field. A collection of such fields overlap to cover the entire visual area.

## Facenet
FaceNet is the name of the facial recognition system that was proposed by Google Researchers in 2015 in the paper titled FaceNet: A Unified Embedding for Face Recognition and Clustering. It achieved state-of-the-art results in the many benchmark face recognition dataset such as Labeled Faces in the Wild (LFW) and Youtube Face Database.
They proposed an approach in which it generates a high-quality face mapping from the images using deep learning architectures such as ZF-Net and Inception. Then it used a method called triplet loss as a loss function to train this architecture. Let’s look at the architecture in more detail.

<img height="250" alt="accuracy" src="https://github.com/gcmadhan/Face-ID-from-webcam-or-Video/blob/main/Readme/deep-learning-architecture.png">

## Features
1. New data set can be added under /train folder with with appropriate folder name
2. Any number or classification can be included, by running the create_model.py Tensorflow model will be created. 
3. Confusion Matrix, Classification report, Loss & Accuracy report will be generated from model creation. 

 
## Ouput
 <img height="250" alt="accuracy" src="https://github.com/gcmadhan/Face-ID-from-webcam-or-Video/blob/main/Readme/Expi-2.gif">


## Limitations/Future changes
1. Training dataset - Images should be in good quality, Images should have single image to categorize it accordingly. 
2. Model has 87% confidence level on the given dataset, any new addition in the dataset might need to create the model again. 
3. Negative scenarios like testing with radom images is not considered. as the categorical will work only based on give datasets. 





