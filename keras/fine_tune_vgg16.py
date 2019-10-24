import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import *
from keras.layers.pooling import *
from keras.preprocessing.image import ImageDataGenerator

model_vgg16 = keras.applications.vgg16.VGG16()

model_vgg16.summary()
print(type(model_vgg16)) # Model
#convert Model to Sequential
model = Sequential()
for layer in model_vgg16.layers:
    model.add(layer)

model.layers.pop() # remove output layer (vgg16 has 1000 outputs)
for layer in model.layers:
    layer.trainable = False # freeze all vgg16 weights
model.add(Dense(2, activation="softmax"))