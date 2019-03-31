#!/usr/bin/env python
# coding: utf-8

# In[1]:


data = "../../coral_img/"


# In[2]:


# Imports
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy
import math
import pandas as pd
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, merge
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
import keras.backend as K
from natsort import natsorted, ns
import keras
import PIL
from keras.regularizers import l2
from glob import glob
from keras.applications import vgg16
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from matplotlib import pyplot


# In[3]:


def brightness_fix(img):
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_YCrCb = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCrCb)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    cl1 = clahe.apply(img_YCrCb[:,:,0])
    
    img_YCrCb[:,:,0] = cl1 
    
    return cv2.cvtColor(img_YCrCb, cv2.COLOR_YCrCb2RGB)


# In[4]:


# Fijar las semillas
seed = 2032 # Semilla del numpy
tf.set_random_seed(seed)# Fijar semilla del keras/tensorflow

epochs = 1000
batch_size = 32

model_name = 'model.h5'


# In[5]:


# Train and test directories
train_dir = data + "train/"
test_dir = data + "Test_Mixed/"


# In[6]:


# Classes
clases = sorted(os.listdir(train_dir))
print(clases)

x_train = np.array([brightness_fix(cv2.imread(os.path.join(train_dir, cl, name), cv2.IMREAD_COLOR)) for cl in clases
           for name in os.listdir(os.path.join(train_dir, cl))])
y_lab = np.array([n for n, cl in enumerate(clases)
           for name in os.listdir(os.path.join(train_dir, cl))])

idx = np.random.permutation(len(x_train))
x_train, y_lab = x_train[idx], y_lab[idx]

test_files = natsorted(os.listdir(test_dir))

x_test = np.array([brightness_fix(cv2.imread(os.path.join(test_dir, name), cv2.IMREAD_COLOR)) 
                   for name in test_files])


# In[7]:



x_train = x_train.astype('float32')
x_train /= 255

x_test = x_test.astype('float32')
x_test /= 255

y_train = to_categorical(y_lab, dtype=int)

print(x_train.shape)
print(len(clases))
print(y_train.shape)


# In[8]:


datagen = ImageDataGenerator(
    shear_range=0.05,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)


# In[9]:


datagen.fit(x_train)

#os.makedirs('images')

for X_batch, y_batch in datagen.flow(x_train, y_train, save_to_dir="images/", save_format="png", save_prefix="aug"):
    for i in range(0, 9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(X_batch[i])
    pyplot.show()
    break


# In[10]:


# create the base pre-trained model
base_model = vgg16.VGG16(weights='imagenet',include_top=False, input_shape=(256,256,3))


# In[11]:


base_model.summary()


# In[12]:


x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dropout(rate=0.25)(x)
x=Dense(1024, activation='relu')(x)
x=Dropout(rate=0.5)(x)
x=Dense(512,activation='relu')(x) 
preds=Dense(14,activation='softmax')(x) 

model=Model(inputs=base_model.input, outputs=preds)


# In[13]:


for i,layer in enumerate(model.layers):
  print(i,layer.name)


# In[14]:


# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in model.layers[:11]:
    layer.trainable=False
for layer in model.layers[11:]:
    layer.trainable=True

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-5),
              metrics=['accuracy'])

model.summary()


# In[17]:


es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=10)


# In[18]:


# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

#file_path="weights.best.hdf5"
#checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size, subset="training"),
                   steps_per_epoch=int(len(x_train)*0.8) / batch_size, epochs=epochs,
                   validation_data = datagen.flow(x_train, y_train, batch_size=batch_size, subset='validation'),
                   validation_steps = int(len(x_train)*0.2) / batch_size,
                   verbose=1, callbacks=[es])

model.save(model_name)


# In[ ]:


predictions_test = model.predict(x_test, verbose=1)

data = pd.DataFrame()
data['Id'] = test_files
data['Category'] = predictions_test.argmax(axis=-1)


data.to_csv("envio.csv", index=False)


# In[ ]:





# In[19]:


conversion = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7,
    'i': 8,
    'j': 9,
    'k': 10,
    'l': 11,
    'm': 12,
    'n': 13
}
classes = []
for t in test_files:
  class_ = t.split("T")[0]
  classes.append(conversion[class_])

from sklearn.metrics import accuracy_score
print(accuracy_score(classes, data['Category']))


# In[ ]:





# In[ ]:




