import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from random import randint
from sklearn.preprocessing import MinMaxScaler
import numpy as np

#generate random data
train_labels = []
train_samples = []
for i in range(1000):
    random_young = randint(13,65)
    train_samples.append(random_young)
    train_labels.append(0)
    
    random_old = randint(65,100)
    train_samples.append(random_old)
    train_labels.append(1)
for i in range(50):
    random_young = randint(13,65)
    train_samples.append(random_young)
    train_labels.append(1)
    
    random_old = randint(65,100)
    train_samples.append(random_old)
    train_labels.append(0)
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
scaled_train_samples = MinMaxScaler(feature_range=(0,1)).fit_transform((train_samples).reshape(-1,1))
test_samples = []
for i in range(1000):
    rand = randint(13,100)
    test_samples.append(rand)
test_samples = np.array(test_samples)
scaled_test_samples = MinMaxScaler(feature_range=(0,1)).fit_transform((test_samples).reshape(-1,1))

#define model net
model = Sequential([
    Dense(16, input_shape=(1,), activation='relu'), #first hidden layer
    Dense(32, activation='relu'), #second hidden layer
    Dense(2, activation='softmax') #output layer
])
#compile with optimiser and loss function
model.compile(
    Adam(lr=.0001), #based on SGD
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)
#train
model.fit(
    scaled_train_samples, 
    train_labels,
    validation_split=0.20, # save 20% for validation # or validation_data for separate set
    batch_size=10, # number of samples used per epoch
    epochs=20, # number of passes
    shuffle=True, 
    verbose=2
)

predictions = model.predict(
    scaled_test_samples, 
    batch_size=10, 
    verbose=0
)

print(predictions)