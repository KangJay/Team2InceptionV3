"""
Found a simple implementation @ https://www.pluralsight.com/guides/introduction-to-densenet-with-tensorflow
Recreated
"""
import tensorflow 

import pandas as pd
import numpy as np
import os
import keras
import random
import cv2
import math
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Convolution2D,BatchNormalization
from tensorflow.keras.layers import Flatten,MaxPooling2D,Dropout
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow import keras
from tensorflow.keras import layers
from numpy import save

import warnings
warnings.filterwarnings("ignore")

import os

model_d = DenseNet121(weights="imagenet", include_top=False,
                    input_shape=(128,128,3))

x=model_d.output

x= GlobalAveragePooling2D()(x)
x= BatchNormalization()(x)
x= Dropout(0.5)(x)
x= Dense(1024,activation='relu')(x) 
x= Dense(512,activation='relu')(x) 
x= BatchNormalization()(x)
x= Dropout(0.5)(x)

preds=Dense(200,activation='softmax')(x) #FC-layer
model=Model(inputs=model_d.input,outputs=preds)

# Freeze layers except last 8
for layer in model.layers[:-8]:
    layer.trainable=False
    
for layer in model.layers[-8:]:
    layer.trainable=True

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Should have a much smaller number of params. try model.summary()

# Get labels and data
data, labels = [], []
subpath = "./tiny-imagenet-200/train/"
"""
def iterate_dirs(path, subdir):
    for filename in os.listdir(path):
        filePath = path + "/" + filename
        if (os.path.isdir(filePath)):
            print(filePath)
            tempSubdir = ""
            if subdir: tempSubdir = subdir + "/" + filename
            else: tempSubdir = filename
            iterate_dirs(filePath, tempSubdir)
"""
# Helper method for recursive 
def iterate_dirs():
    image_paths, labels = np.empty(0), np.empty(0) 
    image_labels = os.listdir(os.getcwd() + "/tiny-imagenet-200/train")
    random.shuffle(image_labels) # shuffle to avoid possible bias in structure alone
    num_class = 0
    for img in image_labels:
        images = sorted(os.listdir(subpath + img + "/images/"))
        #print(len(images))
        for image in images:
            image = f"{subpath}{img}/images/{image}"
            image_paths = np.append(image_paths, image)
            labels = np.append(labels, img)
            #print(labels)
            """for image in images:
            image = cv2.imread(f"{subpath}{img}/images/{image}")
            image = cv2.resize(image, (128, 128))
            image = img_to_array(image)
            data.append(image)
            labels.append(img)"""
        print(f"Done with {img}. File {num_class}/200")
        num_class += 1
    #print(image_paths.shape)
    #print(labels.shape)
    from sklearn.utils import shuffle
    images_shuffled, labels_shuffled = shuffle(image_paths, labels)
    save('./binary-files/images.npy', images_shuffled)
    save('./binary-files/labels.npy', labels_shuffled)

if __name__ == "__main__":
    iterate_dirs()
