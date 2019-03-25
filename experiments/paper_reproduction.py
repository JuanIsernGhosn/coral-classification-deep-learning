import numpy as np 
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.core import Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import keras
from keras.optimizers import SGD
print(K.tensorflow_backend._get_available_gpus())

# Global parameters
epochs = 700

def model_(w=256, h=256, output_size=14):
    model = Sequential()
    model.add(Conv2D(64, 3, 3, input_shape=(w,h, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(output_size, activation='softmax'))
    model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

def VGG_16(weights_path=None, w=256, h=256, output_size=14):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(w,h,3)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_size, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)
    model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model
    

 # Train and test directories
train_dir = "coralspecies/train/train/"
test_dir = "coralspecies/Test_Mixed/Test_Mixed/"
# Classes
clases = sorted(os.listdir(train_dir))

x_train = np.array([cv2.imread(os.path.join(train_dir, cl, name)) for cl in clases
           for name in os.listdir(os.path.join(train_dir, cl))])
y_train = np.array([cl for cl in clases
           for name in os.listdir(os.path.join(train_dir, cl))])
print(x_train.shape)
# print(len(clases))
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
encoder = LabelBinarizer()
y_train = encoder.fit_transform(y_train)
print(y_train.shape)
#import pdb; pdb.set_trace()
# Create partitions inside 
#X_train, X_val, y_train, y_val = train_test_split(
#                                    x_train, y_train, test_size=0.2,
#                                    random_state=99)


# Data Augmentation
datagen = ImageDataGenerator(
    # featurewise_center=True,
    # featurewise_std_normalization=True,
    rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)

# model = VGG_16(weights_path='vgg16_weights.h5')
# Let's try inception model
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(14, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=SGD(lr=0.001, momentum=0.9, decay=1e-6), 
              loss='categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import EarlyStopping 

es = EarlyStopping(monitor='val_acc', mode='max', min_delta=1, patience=8)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)


# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=8),
                   steps_per_epoch=len(x_train) / 8, epochs=epochs,
                   callbacks=[es])

#model.fit(x=x_train, y=y_train, epochs=40, verbose=1, 
    #validation_data=None, 
#    shuffle=True, batch_size=5)
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers


model.save('second_try_inception_v3.h5')

'''
from keras.models import load_model
model = load_model('firt_try_inception_v3.h5')
'''

x_test = []
import glob
for name in glob.glob(test_dir+"/*.jpg"):
    x_test.append(cv2.imread(name))

x_test = np.asarray(x_test)
predictions_test = model.predict(x_test, verbose=1)
y_classes = predictions_test.argmax(axis=-1)
import pandas as pd

names = glob.glob(test_dir+"/*.jpg")
id_ = []
for n in names:
    id_.append(n.split("/")[-1])
data = pd.DataFrame()

data['Id'] = id_
data['Category'] = y_classes

data.to_csv("envio.csv", index=False)
