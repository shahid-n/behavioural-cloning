#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 22:18:44 2020

@author: shahid
"""

import csv, numpy as np, os, scipy.ndimage as ndimg, sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential, load_model
from keras.layers import AveragePooling2D, Cropping2D, ELU, Lambda
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras import optimizers

def instantiate_model(learningRate):
    ch, row, col = 3, 160, 320  # Original image format
    
    model = Sequential()
    # Pre-process incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(row,col,ch)))
    # Crop the image to focus on the region of interest
    model.add(Cropping2D(cropping=((69,25), (0,0))))
    model.add(Conv2D(24, kernel_size=(5, 5)))
    model.add(ELU())
#     model.add(Conv2D(36, kernel_size=(5, 5)))
#     model.add(ELU())
    model.add(Conv2D(48, kernel_size=(5, 5)))
    model.add(ELU())
#     model.add(Conv2D(64, kernel_size=(3, 3)))
#     model.add(ELU())
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(ELU())
    
    model.add(Flatten())
    model.add(Dense(100))
    model.add(ELU())
    model.add(Dropout(0.25))
#     model.add(Dense(50))
#     model.add(ELU())
#     model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(ELU())
    model.add(Dense(1))
  
    adamOpt = optimizers.Adam(lr=learningRate)
    model.compile(loss='mse', optimizer=adamOpt)
    
    return model

def append_data(images, angles, batchSample, steerOffset=0.2):
    # Steering angle offsets for Centre, Left & Right images, respectively
    offset = steerOffset*np.array([0, 1, -1])
    for i in range(len(offset)):
        name = '/opt/carnd_p3/data/IMG/' + batchSample[i].split('/')[-1]
        image = ndimg.imread(name)
        angle = float(batchSample[3]) + offset[i]
        images.append(image)
        angles.append(angle)
        # Now flip the image left-to-right
        flippedImg = np.fliplr(image)
        images.append(flippedImg)
        angles.append(-angle)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            # Tuneable angle correction
            steer_angle_offset = 0.2
            for batch_sample in batch_samples:
                append_data(images, angles, batch_sample, steer_angle_offset)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

samples = []
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Set the batch size
batchSize = 32

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batchSize)
validation_generator = generator(validation_samples, batch_size=batchSize)

print('\nLoading model ...\n\n')
learningRate = 0.0001
model = instantiate_model(learningRate)

mdlWtsFname = os.path.abspath('./model_weights.h5')
mdlFname = os.path.abspath('./model.h5')
if os.path.isfile(mdlWtsFname):
    model.load_weights(mdlWtsFname)

print('Training ...\n')

model.fit_generator(train_generator,
            steps_per_epoch=np.ceil(len(train_samples)/batchSize),
            validation_data=validation_generator,
            validation_steps=np.ceil(len(validation_samples)/batchSize),
            epochs=7, verbose=1)

model.save(mdlFname)
model.save_weights(mdlWtsFname)
print('\n\nModel weights have been saved to the path below.\n' + mdlWtsFname)