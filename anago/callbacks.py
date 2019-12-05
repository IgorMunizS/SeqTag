"""
Custom callbacks.
"""
import numpy as np
from keras.callbacks import Callback
from seqeval.metrics import f1_score, classification_report
from sklearn.metrics import balanced_accuracy_score, f1_score as f1, classification_report as clsrep
import os

class F1score(Callback):

    def __init__(self, seq, preprocessor=None, fold=0):
        super(F1score, self).__init__()
        self.seq = seq
        self.p = preprocessor
        self.best_bacc = 0
        self.fold = fold

    def model_checkpoint(self, bacc, epoch):
        if bacc > self.best_bacc:
            # remove previous checkpoints to save space

            self.best_bacc = bacc
            self.model.save('../models/best_model_' + '_' + str(self.fold) + '.h5')
            print(f"\n{'# ' *20}\nSaved new checkpoint\n{'# ' *20}\n")

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
        label_true = []
        label_pred = []
        for i in range(len(self.seq)):
            x_true, y_true = self.seq[i]
            lengths = self.get_lengths(y_true)
            y_pred = self.model.predict_on_batch(x_true)

            y_true = self.p.inverse_transform(y_true, lengths)
            y_pred = self.p.inverse_transform(y_pred, lengths)

            label_true.extend(y_true)
            label_pred.extend(y_pred)

        score = f1_score(label_true, label_pred)
        print(' - f1: {:04.2f}'.format(score * 100))

        print(classification_report(label_true, label_pred))

        label_true = [item for sublist in label_true for item in sublist]
        label_pred = [item for sublist in label_pred for item in sublist]
        classes = np.unique(label_true)
        # classes2 = np.unique(label_pred)
        #
        # print('Classes: ', classes, classes2)

        bacc = balanced_accuracy_score(label_true, label_pred)
        print(' - BACC: {:04.2f}'.format(bacc * 100))

        score = f1(label_true, label_pred, average='micro')
        print(' - f1: {:04.2f}'.format(score * 100))

        print(clsrep(label_true, label_pred, labels=classes))
        self.model_checkpoint(bacc)
        logs['f1'] = score
