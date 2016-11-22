from __future__ import absolute_import
from __future__ import print_function
import os.path

import csv

import numpy as np
from time import localtime, strftime
import json
import warnings

import keras
from collections import deque, OrderedDict, Iterable
from keras.utils.generic_utils import Progbar
from keras import backend as K
from pkg_resources import parse_version

class CSVHistory(keras.callbacks.Callback):
    '''Callback that streams epoch results to a csv file.
    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.
    # Example
        ```python
            csv_logger = CSVLogger('training.log')
            model.fit(X_train, Y_train, callbacks=[csv_logger])
        ```
    Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        modelname: the name
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    '''

    def __init__(self, filename, modelname, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.modelname = modelname
        self.append = append
        self.writer = None
        self.keys = None
        self.file_exists = os.path.isfile(self.filename)
        super(CSVHistory, self).__init__()

    def _set_model(self, model):
        self.model = model

    def on_train_begin(self, logs={}):
        if self.append:
            self.csv_file = open(self.filename, 'a')
            #print "csv file opened"
        else:
            self.csv_file = open(self.filename, 'w')
            #print "csv file opened"

    def on_epoch_end(self, epoch, logs={}):
        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(lambda x: str(x), k)))
            else:
                return k

        self.current_time = strftime("%d %b %Y %H:%M:%S", localtime())

        if not self.writer:
            self.keys = sorted(logs.keys())
            self.writer = csv.DictWriter(self.csv_file, fieldnames= ["epoch", "model", "param_count", 'time', "layers", "optimizer", "loss"]+ self.model.optimizer.get_config().keys() + self.keys)

            if not self.file_exists:
                self.writer.writeheader()

        self.row_dict = OrderedDict({'epoch': epoch,
                               "model": self.modelname,
                               "param_count": self.model.count_params(),
                               "time": self.current_time,
                               "layers": self.model.layers,
                               "optimizer": self.model.optimizer,
                               "loss": self.model.loss})

        self.row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.row_dict.update(self.model.optimizer.get_config())
        self.writer.writerow(self.row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs={}):
        self.csv_file.close()

class LRdecay(keras.callbacks.Callback):
    '''LRdecay
    # Arguments
        DECAY: float (0,1] to be multiplied by the current learn rate when the specified training metric stops improving.
        METRIC: "val_acc" or "val_loss"

    '''
    def __init__(self, decay):
        super(LRdecay, self).__init__()
        self.decay = decay

    def on_epoch_end(self, epoch, logs={}):
        pass
