import keras
from keras import backend as K
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Activation
from keras.layers.core import Dense

#1 load all: architecture, weights, training config (loss, optimizer), state of optimizer
model = load_model("saveAll.h5")
model.summary()
print("Old optimiser:", model.optimizer, "Old weights", model.weights[0], sep="\n") # print only first layer's weights

#2 load only architecture
with open("architecture.json", "r") as file:
    jsonStr = file.read()
model_architecture = model_from_json(jsonStr)
model_architecture.summary()
print("model_architecture with new random weights:", model_architecture.weights[0], sep="\n")

#3 load only weights
#same as before
model2 = Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])
print("model2 with new random weights:", model2.weights[0], sep="\n")
model2.load_weights("weights.h5")
print("model2 after loading old weights:", model2.weights[0], sep="\n")
