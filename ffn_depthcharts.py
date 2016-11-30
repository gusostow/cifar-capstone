import numpy as np
import json

import keras
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.regularizers import l2, activity_l2
from keras.layers.normalization import BatchNormalization

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Activation
from keras.layers import Dense
from keras.layers import Dropout
from custom_augmentation import image_alt

#custom csv logger callback
from custom_callbacks.customcalls import CSVHistory
from keras.callbacks import ModelCheckpoint

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#normalize images
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train = X_train / 255.0
X_test = X_test / 255.0


#reshape images to vectors
X_train = np.reshape(X_train, (50000, 3072))
X_test = np.reshape(X_test, (10000, 3072))

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

model_0, model_1, model_2, model_3, model_4, model_5, model_6 = Sequential()

models = [model_0, model_1, model_2, model_3, model_4, model_5, model_6]
modelnames = ["model_0", "model_1", "model_2", "model_3", "model_4", "model_5","model_6"]

model_tuples = zip(models, modelnames)

width = 512

#define model_0
model_0.add(Dense(num_classes, input_dim=3072, activation = "softmax"))

#define model_1
model_1.add(Dense(width, input_dim=3072))
model_1.add(Dense(num_classes), activation = "softmax")

#define model_2
model_2.add(Dense(width, input_dim=3072, activation = "relu"))
model_2.add(Dense(width), activation = "relu")
model_2.add(Dense(num_classes), activation = "softmax")

#define model_3
model_3.add(Dense(width, input_dim=3072, activation = "relu"))
model_3.add(Dense(width), activation = "relu")
model_3.add(Dense(width), activation = "relu")
model_3.add(Dense(num_classes), activation = "softmax")

#define model_4
model_4.add(Dense(width, input_dim=3072, activation = "relu"))
model_4.add(Dense(width), activation = "relu")
model_4.add(Dense(width), activation = "relu")
model_4.add(Dense(width), activation = "relu")
model_4.add(Dense(num_classes), activation = "softmax")

#define model_5
model_5.add(Dense(width, input_dim=3072, activation = "relu"))
model_5.add(Dense(width), activation = "relu")
model_5.add(Dense(width), activation = "relu")
model_5.add(Dense(width), activation = "relu")
model_5.add(Dense(num_classes), activation = "softmax")

#define model_6
model_6.add(Dense(width, input_dim=3072, activation = "relu"))
model_6.add(Dense(width), activation = "relu")
model_6.add(Dense(width), activation = "relu")
model_6.add(Dense(width), activation = "relu")
model_6.add(Dense(width), activation = "relu")
model_6.add(Dense(num_classes), activation = "softmax")

# COMPILE
epochs = 20
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, decay = decay,momentum = 0.9, nesterov=False)

for models, modelname in model_tuples:
    csv = CSVHistory("csv_logs/ffn_depthcharts.csv", modelname, separator = " , ", append = False)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    print "now training", modelname

    fit = model.fit(X_train, y_train, validation_data = (X_test, y_test), nb_epoch=epochs, batch_size=32, shuffle = True, callbacks = [csv])


