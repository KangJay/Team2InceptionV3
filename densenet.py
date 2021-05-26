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
#from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.utils import Sequence, to_categorical
from keras import callbacks
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
                    input_shape=(None, None, 3)) # (None, None, 3) = Can accept any input size

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

model.compile(optimizer='Adam',loss='categorical_crossentropy',
              metrics=['accuracy'])

# Prepping data


# Constants
target_img_size=(96, 96) # Can change
seed = 0

#1, 2, 3, 4 are temporary column names. These aren't needed and are dropped.
val_data = pd.read_csv("./tiny-imagenet-200/val/val_annotations.txt", sep="\t",
                  header=None, names=['file', 'class', '1', '2', '3', '4'])
val_data.drop(['1', '2', '3', '4'], axis=1, inplace=True)

# Data Generators
train_datagen = ImageDataGenerator(rescale=1.0/255) # Normalizing images (RGB values = 0 to 255)
valid_datagen = ImageDataGenerator(rescale=1.0/255)

#Keep seed as 0 so we can consistently compare
train_data_generator = train_datagen.flow_from_directory("./tiny-imagenet-200/train/",
                            target_size=target_img_size, color_mode="rgb", 
                            batch_size=64, class_mode="categorical",
                            shuffle=True, seed=seed)
validation_data_generator = valid_datagen.flow_from_dataframe(val_data,
                                directory="./tiny-imagenet-200/val/images", 
                                x_col="file", y_col="class",
                                target_size=target_img_size, color_mode="rgb",
                                class_mode="categorical", batch_size=64,
                                shuffle=True, seed=seed)

earlystop = callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5,
                                    restore_best_weights=True)
# earlystop = callbacks.ReduceLROnPlateau(mode="min")

model.fit_generator(train_data_generator, epochs=15, steps_per_epoch=200,
                    validation_steps=200,
                    validation_data=validation_data_generator, verbose=1,
                    callbacks=[earlystop])
model.save("./models/first_model.h5")



