"""
Custom callbacks.
"""
import numpy as np
from keras.callbacks import Callback
from seqeval.metrics import f1_score, classification_report
from sklearn.metrics import balanced_accuracy_score
from utils.utils import inverse_transform

class BACCscore(Callback):

    def __init__(self, val_data, batch_size=20, label_encoder=None):
        super().__init__()
        self.validation_data = val_data
        self.batch_size = batch_size
        self.label_encoder = label_encoder


    def get_lengths(self, y_true):
        lengths = []
        for y in np.argmax(y_true, -1):
            try:
                i = list(y).index(0)
            except ValueError:
                i = len(y)
            lengths.append(i)

        return lengths

    def on_epoch_end(self, epoch, logs={}):
        batches = len(self.validation_data)
        total = batches * self.batch_size

        val_pred = np.zeros((total, 1))
        val_true = np.zeros((total))

        for batch in range(batches):
            xVal, yVal = self.validation_data.__getitem__(batch)
            val_pred[batch * self.batch_size: (batch + 1) * self.batch_size] = np.asarray(
                self.model.predict(xVal)).round()
            val_true[batch * self.batch_size: (batch + 1) * self.batch_size] = yVal

        label_true = []
        label_pred = []
        for i in range(len(val_pred)):
            y_pred, y_true = val_pred[i], val_true[i]
            lengths = self.get_lengths(y_true)

            y_true = inverse_transform(y_true, lengths)
            y_pred = inverse_transform(y_pred, lengths)

            label_true.extend(y_true)
            label_pred.extend(y_pred)

            y_true = inverse_transform(y_true, self.label_encoder, lengths)
            y_pred = inverse_transform(y_pred, self.label_encoder, lengths)

            label_true.extend(y_true)
            label_pred.extend(y_pred)

        score = balanced_accuracy_score(label_true, label_pred)
        print(' - BACC: {:04.2f}'.format(score * 100))
        score = f1_score(label_true, label_pred)
        print(' - f1: {:04.2f}'.format(score * 100))


class Metrics(Callback):

    def __init__(self, val_data, batch_size=20):
        super().__init__()
        self.validation_data = val_data
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        print(self.validation_data)
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        batches = len(self.validation_data)
        total = batches * self.batch_size

        val_pred = np.zeros((total, 1))
        val_true = np.zeros((total))

        for batch in range(batches):
            xVal, yVal = next(self.validation_data)
            val_pred[batch * self.batch_size: (batch + 1) * self.batch_size] = np.asarray(
                self.model.predict(xVal)).round()
            val_true[batch * self.batch_size: (batch + 1) * self.batch_size] = yVal

        val_pred = np.squeeze(val_pred)
        _val_f1 = f1_score(val_true, val_pred)
        _val_precision = precision_score(val_true, val_pred)
        _val_recall = recall_score(val_true, val_pred)

        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)

        return