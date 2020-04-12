#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 22:18:44 2020

@author: shahid
"""

import csv, numpy as np, os, PIL.Image as pimg, sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential, load_model
from keras.layers import AveragePooling2D, Cropping2D, Lambda
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import bespokeLoss as BL

import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

def instantiate_model(learningRate):
    ch, row, col = 3, 160, 320  # Original image format
    
    model = Sequential()
    # Crop the image to focus on the region of interest
    model.add(Cropping2D(cropping=((46,24), (0,0)), input_shape=(row,col,ch)))
    # Downsample the image further using average pooling
    model.add(AveragePooling2D(pool_size=(2, 2)))
    # Normalise incoming data, centred around zero within a small rational interval [-1, 1] 
    model.add(Lambda(lambda x: x/127.5 - 1.0))
    # Convolutional layers
    model.add(Conv2D(24, kernel_size=(5, 5)))
    model.add(Conv2D(36, kernel_size=(5, 5)))
    model.add(Conv2D(48, kernel_size=(5, 5)))
    model.add(Conv2D(44, kernel_size=(5, 5)))
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(Conv2D(64, kernel_size=(3, 3)))
    # Dense layers
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    
    adamOpt = optimizers.Adam(lr=learningRate)
    model.compile(loss=BL.bespoke_loss, optimizer=adamOpt)
    
    return model

def append_data(images, angles, batchSample, steerOffset=0.2):
    # Steering angle offsets for Centre, Left & Right images, respectively
    offset = steerOffset*np.array([0, 1, -1])
    for i in range(len(offset)):
        name = './my_data/IMG/' + batchSample[i].split('/')[-1]
        img = pimg.open(name)
        image = np.asarray(img)
        angle = float(batchSample[3]) + offset[i]
        images.append(image)
        angles.append(angle)
        # Now flip the image left-to-right
        flippedImg = np.fliplr(image)
        images.append(flippedImg)
        angles.append(-angle)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    # Tuneable angle correction
    steer_angle_offset = 0.2
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                append_data(images, angles, batch_sample, steer_angle_offset)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

samples = []
with open('./my_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Discard the header row
samples = samples[1:]

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Set the batch size
batchSize = 32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batchSize)
validation_generator = generator(validation_samples, batch_size=batchSize)

print('\nLoading model ...\n\n')
learningRate = 0.00001
# learningRate = 0.001
model = instantiate_model(learningRate)

mdlWtsFname = os.path.abspath('./best_weights.h5')
mdlWtsSaveName = os.path.abspath('./model_weights.h5')
mdlFname = os.path.abspath('./model.h5')
if os.path.isfile(mdlWtsFname):
    model.load_weights(mdlWtsFname)

print('Training ...\n')

chkPt = ModelCheckpoint("./best_model.h5", monitor='val_loss', verbose=1,
    save_best_only=True, mode='auto', period=1)
chkPtWts = ModelCheckpoint("./best_weights.h5", monitor='val_loss', verbose=1,
    save_best_only=True, save_weights_only=True, mode='auto', period=1)

model.fit_generator(train_generator,
            steps_per_epoch=np.ceil(len(train_samples)/batchSize),
            validation_data=validation_generator,
            validation_steps=np.ceil(len(validation_samples)/batchSize),
            epochs=12, verbose=1, callbacks=[chkPt, chkPtWts])

model.save(mdlFname)
model.save_weights(mdlWtsSaveName)
print('\n\nModel weights have been saved to the path below.\n' + mdlWtsSaveName)
