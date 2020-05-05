# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 19:14:13 2020

@author: ed2049
"""

# Imports:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
from PIL import Image

#Settings:
image_size = 1288
n_images = 5000

# Functions:
"Flatten RGB image------------------------------------------------------------"
def flatten(img):
    flat = img.T.flatten().reshape(1, -1)
    img = np.reshape(flat,(3,image_size, image_size))
    img = np.moveaxis(img, 0,2)
    flat = img.T.flatten().reshape(1, -1)
    return flat

# Read labels:
read_data = pd.read_excel('labels.xlsx')
data = read_data[['idImage','Image_Name','Database','Classification']]

data = data.dropna(how = 'any') # drop images withot classification
data['Classification'] = data['Classification'].apply(lambda x: int(x)) # convert labels to ints
data = data.loc[(data['Classification'] >= 0) & (data['Classification'] < 3)] # remove defective images (label = -1) and cat 3

data = data.loc[data['idImage'] <= n_images] # get the selected number of images

# Implement train-validation-test split:
train, val_test = train_test_split(data, test_size = 0.4, random_state = 10)
validation, test = train_test_split(val_test, test_size = 0.5, random_state = 10)

# Check the percentage of cat. 2 images in train, validation and test split:
cat2_in_train = len(train.loc[train['Classification'] == 2]['Classification'])
len_train = len(train['Classification'])
perc_in_train = (100*cat2_in_train)/len_train

print('%.2f percent of cat. 2 in Train'%perc_in_train)

cat2_in_validation = len(validation.loc[validation['Classification'] == 2]['Classification'])
len_validation = len(validation['Classification'])
perc_in_validation = (100*cat2_in_validation)/len_validation

print('%.2f percent of cat. 2 in Validation'%perc_in_validation)

cat2_in_test = len(test.loc[test['Classification'] == 2]['Classification'])
len_test = len(test['Classification'])
perc_in_test = (100*cat2_in_test)/len_test

print('%.2f percent of cat. 2 in Test'%perc_in_test)
print('%i Images'%(len_train + len_validation + len_test))

# Reset index of DataFrames:
train = train.reset_index(drop = True)
validation = validation.reset_index(drop = True)
test = test.reset_index(drop = True)

# Get the labels of the images:
train_labels = train['Classification']
validation_labels = validation['Classification']
test_labels = test['Classification']

# Get the image names:
train_names = train['Image_Name']
validation_names = validation['Image_Name']
test_names = test['Image_Name']

# Read Images and convert them to correct format:    
path = 'Images/'


img1_train = Image.open(path+str(train_names[0])+".jpeg") #Open image
training_images = flatten(np.array(img1_train)) #Create np array with flat image

img1_validation = Image.open(path+str(validation_names[0])+".jpeg") #Open image
validation_images = flatten(np.array(img1_validation)) #Create np array with flat image


# Collect all trainning images:
n = 0
for i in train_names[1:]:
    img = Image.open(path+str(i)+".jpeg") #Open image
    img_flat = flatten(np.array(img))
    training_images = np.vstack([training_images, img_flat]) #stack the image in the array
    print('Processing image %i / %i of the train set'%(n, len(train_names) - 1))
    n += 1

# Collect all validation images:
n = 0
for i in validation_names[1:]:
    img = Image.open(path+str(i)+".jpeg") #Open image
    img_flat = flatten(np.array(img))
    validation_images = np.vstack([validation_images, img_flat]) #stack the image in the array
    print('Processing image %i / %i of the validation set'%(n, len(validation_names) - 1))
    n += 1
    
    


'''
################### Get images with np.empty ################
# Collect all trainning images:
img = Image.open(path+str(train_names[0])+".jpeg") #Open image
img_flat = flatten(np.array(img))

training_images = np.empty([len(train_names), img_flat.shape[1]])
n = 0
for i in train_names[0:]:
    img = Image.open(path+str(i)+".jpeg") #Open image
    img_flat = flatten(np.array(img))
    training_images[n, :] = img_flat
    print('Processing image %i / %i of the train set'%(n, len(train_names) - 1))
    n += 1
    
# Collect all validation images:
validation_images = np.empty([len(validation_names), img_flat.shape[1]])
n = 0
for i in validation_names[0:]:
    img = Image.open(path+str(i)+".jpeg") #Open image
    img_flat = flatten(np.array(img))
    validation_images[n, :] = img_flat
    print('Processing image %i / %i of the validation set'%(n, len(validation_names) - 1))
    n += 1
#############################################################
'''    
    
Training_images = training_images.reshape(len(train_labels), 3, image_size, image_size).transpose(0,2,3,1)/255
Validation_images = validation_images.reshape(len(validation_labels), 3, image_size, image_size).transpose(0,2,3,1)/255
print('\nImages reshaped successfully')

"================Create Convolutional Neural Networks Model==================="

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from livelossplot import PlotLossesKeras
import keras_metrics

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)

t1 = time.clock()

"============================ CNN SETTINGS ===================================="
Batch_Size = 5
Epochs = 50
Learning_Rate = 0.0001
Dropout1 = 0.5
Dropout2 = 0.6
Dropout3 = 0.7
Dense_Units1 = 64
Reg = 0.001
Pooling = 3
"============================================================================="

batch_size = Batch_Size
num_classes = 3 #Three different categories
epochs = Epochs

# input image dimensions
img_rows, img_cols = image_size, image_size

# the data, split between train and validation sets
x_train = Training_images
y_train = train_labels
x_validation = Validation_images
y_validation = validation_labels

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_validation = keras.utils.to_categorical(y_validation, num_classes)

"==================================Model CNN=================================="
print('\nTrainning CNN')

model = Sequential()

#Convolution:
model.add(Conv2D(32, kernel_size = (3, 3),
                 padding = 'same',
                 activation = 'relu',
                 input_shape = (image_size, image_size, 3)))

#Convolution:
model.add(Conv2D(32, (3, 3), activation = 'relu'))

#Pool:
model.add(MaxPooling2D(pool_size = (Pooling,Pooling)))
model.add(Dropout(Dropout1))

#Convolution:
model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))

#Convolution:
model.add(Conv2D(64, (3, 3), activation = 'relu'))

#Pool:
model.add(MaxPooling2D(pool_size = (Pooling, Pooling)))
model.add(Dropout(Dropout2))

#Flatten:
model.add(Flatten())

'''
##Dense1:
#model.add(Dense(Dense_Units1,activation='relu'))
#model.add(Dropout(Dropout3))
'''

#Dense:
model.add(Dense(num_classes, activation = 'softmax'))

#Define Metrics:
precision = keras_metrics.binary_precision(label = 2)
recall = keras_metrics.binary_recall(label = 2)

model.compile(loss = keras.losses.categorical_crossentropy,
              optimizer = keras.optimizers.Adamax(lr = Learning_Rate,
                                                  beta_1 = 0.9,
                                                  beta_2 = 0.999,
                                                  epsilon = 0.00000001,
                                                  decay = 0.0),
              metrics = ['accuracy',precision,recall])

modeling = model.fit(x_train, y_train,
                     batch_size = batch_size,
                     epochs = epochs,
                     verbose = 1,
                     validation_data = (x_validation, y_validation),
                     callbacks = [PlotLossesKeras()])


############################### Model Test ####################################

img1_test = Image.open(path+str(test_names[0])+".jpeg") # Open image
test_images = flatten(np.array(img1_test)) # Create np array with flat image
n = 0
for i in test_names[1:]:
    img = Image.open(path+str(i)+".jpeg") # Open image
    img_flat = flatten(np.array(img))
    test_images = np.vstack([test_images, img_flat]) #stack the image in the array
    print('Processing image %i / %i of the test set'%(n, len(test_names) - 1))
    n += 1
Test_images = test_images.reshape(len(test_labels), 3, image_size, image_size).transpose(0,2,3,1)/255

x_test = Test_images
y_test = test_labels

# convert class vectors to binary class matrices
y_test = keras.utils.to_categorical(y_test, num_classes)

''' 
#test_predictions = model.predict(x_test, batch_size = batch_size)
#test_predictions = pd.DataFrame(test_predictions, columns = ['pr_0', 'pr_1', 'pr_2'])
#def evaluate_prediction(row):
#    if row['pr_0'] > row['pr_1'] and row['pr_0'] > row['pr_2']:
#        return 0
#    elif row['pr_1'] > row['pr_0'] and row['pr_1'] > row['pr_2']:
#        return 1
#    else:
#        return 2    
#test_predictions['label'] = test_predictions.apply(lambda row: evaluate_prediction(row), axis = 1)
'''
 
# Implement the score in the test data:
score = model.evaluate(x_test, y_test, verbose = 0, batch_size = batch_size,)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Test precision:', score[2])
print('Test recall:', score[3])

#Save Model:
model.save('model_2')

t2 = time.clock() - t1

print('Model trained in %.2f seconds'%t2)