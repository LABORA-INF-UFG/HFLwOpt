import numpy as np
import pandas as pd

from load_data.LoadData import LoadData
from ml_model.MLModel import Model


class Client:
    def __init__(self, cid, load_data_constructor, cm):
        self.cid = int(cid)
        self.load_data_constructor = load_data_constructor
        self.cm = cm

        self.model = Model.create_model(cm=self.cm)
        self.load_data = LoadData(cm=self.cm)

        if self.load_data_constructor:
            (self.x_train, self.y_train), (self.x_test, self.y_test), self.number_samples = self.load_data.data_client(self.cid)
        else:
            (self.x_train, self.y_train), (self.x_test, self.y_test), self.number_samples = (None, None), (None, None), 0

    def number_data_samples(self):
        (self.x_train, _), (_, _), _ = self.load_data.data_client(self.cid)
        return len(self.x_train)

    def fit(self, parameters, config=None):

        if not self.load_data_constructor:
            (self.x_train, self.y_train), (self.x_test, self.y_test), self.number_samples = self.load_data.data_client(self.cid)

        self.model.set_weights(parameters)
        history = self.model.fit(self.x_train,
                                 self.y_train,
                                 epochs=1,
                                 batch_size=128,
                                 validation_data=(self.x_test, self.y_test),
                                 verbose=False)
        sample_size = len(self.x_train)

        if not self.load_data_constructor:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = (None, None), (None, None)

        return self.model.get_weights(), sample_size, {"val_accuracy": history.history['val_accuracy'][-1],
                                                       "val_loss": history.history['val_loss'][-1]}

    def evaluate(self, parameters):
        if not self.load_data_constructor:
            (self.x_train, self.y_train), (self.x_test, self.y_test), self.number_samples = self.load_data.data_client(self.cid)

        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=False)

        if not self.load_data_constructor:
            (self.x_train, self.y_train), (self.x_test, self.y_test), self.number_samples = (None, None), (None, None), 0

        return loss, accuracy
