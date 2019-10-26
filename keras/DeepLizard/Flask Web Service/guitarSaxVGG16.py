import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import *
from keras.layers.pooling import *
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

train_batches = ImageDataGenerator().flow_from_directory(r"C:\Users\rahul\Documents\github\learningML\datasets\guitar-saxophone\train", target_size=(224,224), batch_size=10) # total 60
test_batches = ImageDataGenerator().flow_from_directory(r"C:\Users\rahul\Documents\github\learningML\datasets\guitar-saxophone\test", target_size=(224,224), batch_size=5) # total 5
valid_batches = ImageDataGenerator().flow_from_directory(r"C:\Users\rahul\Documents\github\learningML\datasets\guitar-saxophone\valid", target_size=(224,224), batch_size=10) # total 20

model_vgg16 = keras.applications.vgg16.VGG16()
model = Sequential()
for layer in model_vgg16.layers:
    model.add(layer)
model.layers.pop() # remove output layer (vgg16 has 1000 outputs)
for layer in model.layers:
    layer.trainable = False # freeze all vgg16 weights
model.add(Dense(2, activation="softmax"))

model.compile(
    Adam(lr=.0001),
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

model.fit_generator(train_batches, steps_per_epoch=6, validation_data=valid_batches, validation_steps=2, epochs=5, verbose=2)
predictions = model.predict_generator(test_batches, steps=1, verbose=0)