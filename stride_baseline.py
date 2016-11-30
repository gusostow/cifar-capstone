import numpy as np
import pdb
import json
from contextlib2 import redirect_stdout

import keras
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from custom_augmentation import image_alt

from keras.optimizers import SGD
from keras.regularizers import l2, activity_l2
from keras.layers.normalization import BatchNormalization

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.pooling import GlobalAveragePooling2D
from keras.constraints import maxnorm
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import TensorBoard, LearningRateScheduler
from custom_callbacks.customcalls import CSVHistory

# ***************\\CHANGE MODEL NAME HERE EVERY RUN//***********************
# **************************************************************************
modelname = "strd_7" #used for logging purposes
# **************************************************************************

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#normalize images
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train = X_train / 255.0
X_test = X_test / 255.0

#one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#subset training data
train_sub_ind = np.random.choice(X_train.shape[0], 5000, replace = False)

#SUBSET TRAINING SET
#X_train = X_train[train_sub_ind, :]
#y_train = y_train[train_sub_ind, :]

#CALLBACKS
board = TensorBoard(log_dir="logs/" + modelname, histogram_freq=0, write_graph=True, write_images=False)
csv = CSVHistory("csv_logs/" + modelname + ".csv", modelname, separator = " , ", append = False)

#DEFINE MODEL
model = Sequential()

model.add(Convolution2D(96, 3, 3, border_mode='same', input_shape=X_train.shape[1:]))
model.add(Activation("relu"))
model.add(Convolution2D(96,3,3, border_mode='same', init = "glorot_normal"))
model.add(Activation("relu"))

model.add(MaxPooling2D((3, 3), strides=(2,2)))

model.add(Convolution2D(192,3,3, border_mode='same', init = "glorot_normal"))
model.add(Activation("relu"))
model.add(Convolution2D(192,3,3, border_mode='same', init = "glorot_normal"))
model.add(Activation("relu"))

model.add(MaxPooling2D((3,3), strides=(2,2)))

model.add(Convolution2D(192,3,3, border_mode='same', init = "glorot_normal"))
model.add(Activation("relu"))

model.add(Convolution2D(192,1,1, border_mode='same', init = "glorot_normal"))
model.add(Activation("relu"))
model.add(Convolution2D(10,1,1, border_mode='same', init = "glorot_normal"))
model.add(Activation("relu"))

model.add(GlobalAveragePooling2D())
model.add(Activation("softmax"))


#Learning rate schedule (as per "striving for simplicity" paper)
def scheduler(epoch):

    initial_lr = 0.01
    if epoch <= 200:
        lr = initial_lr
    elif epoch <= 250:
        lr = initial_lr * 0.1
    elif epoch <= 300:
        lr = initial_lr * 0.01
    else:
        lr = initial_lr * 0.001
    return lr

#create callback
change_lr = LearningRateScheduler(scheduler)

# COMPILE
epochs = 100
batch_size = 32

sgd = SGD(lr=0, decay = 0, momentum = 0.9, nesterov=False)

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

data_augmentation = True

print model.summary()

with open("models/" + modelname + ".json", 'wb') as fp:
    json.dump(model.to_json(), fp)

with open("summaries/" + modelname + '.txt', 'w') as f:
    with redirect_stdout(f):
        print model.summary()

if not data_augmentation:
    print 'Not using data augmentation.'
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size= batch_size, callbacks = [board, csv])
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

    #pdb.set_trace()
    model.fit_generator(datagen.flow(X_train, y_train,
                            batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=epochs,
                            validation_data=(X_test, y_test),
                            callbacks = [change_lr, board, csv])

#model.save_weights("weights/" + modelname + ".hdf5")
