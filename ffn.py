import numpy as np
from matplotlib import pyplot
from scipy.misc import toimage
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

# ***************\\CHANGE MODEL NAME HERE EVERY RUN//***********************
# **************************************************************************
modelname = "ffn42" #used for logging purposes
# **************************************************************************


seed = 7
np.random.seed(seed)

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

train_sub_ind = np.random.choice(X_train.shape[0], 25000, replace = False)

#SUBSET TRAINING SET
#X_train = X_train[train_sub_ind, :]
#y_train = y_train[train_sub_ind, :]

"""
************************LOAD A MODEL FROM JSON*************************
***********************************************************************
with open('./models/ffn19.json', 'rb') as fp:
    saved_model = json.load(fp)
model = model_from_json(saved_model)
"""


#define model
model = Sequential()
model.add(Dense(512, input_dim=3072))
#model.add(BatchNormalization())
model.add(Activation("relu"))
#model.add(Dropout(0.02))

model.add(Dense(512))
#model.add(BatchNormalization())
model.add(Activation("relu"))
#model.add(Dropout(0.02))

model.add(Dense(512))
#model.add(BatchNormalization())
model.add(Activation("relu"))
#model.add(Dropout(0.02))

model.add(Dense(10))
#model.add(BatchNormalization())
model.add(Activation("softmax"))


# COMPILE
epochs = 5
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, decay = decay,momentum = 0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#CALLBACKS
#board = keras.callbacks.TensorBoard(log_dir="logs/" + modelname, histogram_freq=0, write_graph=True, write_images=False)
#csv = CSVHistory("csv_logs/log_ffn_adam.csv", modelname, separator = " , ", append = True)

filepath = modelname + "_" + "{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')

data_augmentation = True

if not data_augmentation:
    print 'Not using data augmentation.'
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size= batch_size, callbacks = [checkpoint])
else:
    print 'Using real-time data augmentation.'

    # this will do preprocessing and realtime data augmentation
    datagen = image_alt.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(X_train)

    model.fit_generator(datagen.flow(X_train, y_train,
                            batch_size=32),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=epochs,
                            validation_data=(X_test, y_test),
                            callbacks = [board, csv])

#FIT
fit = model.fit(X_train, y_train, validation_data = (X_test, y_test), nb_epoch=epochs, batch_size=32, shuffle = True, callbacks = [checkpoint])

with open("models/" + modelname + ".json", 'wb') as fp:
    json.dump(model.to_json(), fp)

model.save_weights("weights/" + modelname + ".hdf5")

