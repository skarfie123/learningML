import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, BatchNormalization
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras import regularizers
from random import randint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import itertools
import matplotlib.pyplot as plt
import numpy as np

#generate random data
train_labels = []
train_samples = []
for i in range(50):
    random_young = randint(13,65)
    train_samples.append(random_young)
    train_labels.append(1)
    
    random_old = randint(65,100)
    train_samples.append(random_old)
    train_labels.append(0)
for i in range(1000):
    random_young = randint(13,65)
    train_samples.append(random_young)
    train_labels.append(0)
    
    random_old = randint(65,100)
    train_samples.append(random_old)
    train_labels.append(1)
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
scaled_train_samples = MinMaxScaler(feature_range=(0,1)).fit_transform((train_samples).reshape(-1,1))
test_samples = []

test_labels = []
test_samples = []
for i in range(10):
    random_young = randint(13,65)
    test_samples.append(random_young)
    test_labels.append(1)
    
    random_old = randint(65,100)
    test_samples.append(random_old)
    test_labels.append(0)
for i in range(200):
    random_young = randint(13,65)
    test_samples.append(random_young)
    test_labels.append(0)
    
    random_old = randint(65,100)
    test_samples.append(random_old)
    test_labels.append(1)
test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
scaled_test_samples = MinMaxScaler(feature_range=(0,1)).fit_transform((test_samples).reshape(-1,1))

#define model net
model = Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])
model.summary()
#compile with optimiser and loss function
model.compile(
    Adam(lr=.0001),
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)
#train
model.fit(
    scaled_train_samples, 
    train_labels,
    validation_split=0.20,
    batch_size=10,
    epochs=20,
    shuffle=True, 
    verbose=2
)

classes = model.predict_classes(
    scaled_test_samples, 
    batch_size=10, 
    verbose=0
)

cm = confusion_matrix(test_labels, classes)

# function copied from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

plot_confusion_matrix(test_labels, classes, ["healthy", "ill"], title="Confusion Matrix")
plt.show()