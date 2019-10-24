import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import *
from keras.layers.pooling import *
from keras.preprocessing.image import ImageDataGenerator
train_batches = ImageDataGenerator().flow_from_directory("path/to/train/dataset", target_size=(32,32), batch_size=10)
#test_batches = ImageDataGenerator...
#valid_batches = ImageDataGenerator...

bad_model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', input_shape = (32,32,3)),
    Flatten(),
    Dense(2, activation='softmax')
])
bad_model.fit_generator(train_batches, steps_per_epoch=4, validation_data=valid_batches, validation_steps=4, epochs=5, verbose=2)
predictions = bad_model.predict_generator(test_batches, steps=4, verbose=0)