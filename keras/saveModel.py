import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.metrics import categorical_crossentropy
from keras.layers.core import Dense
from keras.optimizers import Adam

model = Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])
model.compile(
    Adam(lr=.0001),
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)
model.fit(
    [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95], #example data
    [0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1,1], #example labels
    validation_split=0.50,
    batch_size=10,
    epochs=20,
    shuffle=True, 
    verbose=2
)
model.summary()

#1 save all: architecture, weights, training config (loss, optimizer), state of optimizer
model.save("saveAll.h5")

#2 save only architecture
jsonStr = model.to_json()
with open("architecture.json", "w") as file:
    file.write(jsonStr)

#3 save only weights
model.save_weights("weights.h5")
