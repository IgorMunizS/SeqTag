"""
Custom callbacks.
"""
import numpy as np
from keras.callbacks import Callback
from seqeval.metrics import f1_score, classification_report
from sklearn.metrics import balanced_accuracy_score, f1_score as f1, classification_report as clsrep
import os
import keras.backend as K

class F1score(Callback):

    def __init__(self, seq, preprocessor=None, fold=0, swa_epoch=7):
        super(F1score, self).__init__()
        self.seq = seq
        self.p = preprocessor
        self.best_bacc = 0
        self.fold = fold
        self.history = [] # to store per each class and also mean PR AUC
        self.early_stopping_patience = 3
        self.plateau_patience = 2
        self.reduction_rate = 0.2
        self.swa_epoch = swa_epoch
        self.swa_filepath = '../models/best_model_' + str(self.fold) + '_swa' + '.h5'
        self.swa_control = 0

    def is_patience_lost(self, patience):
        if len(self.history) > patience:
            best_performance = max(self.history[-(patience + 1):-1])
            return best_performance == self.history[-(patience + 1)] and best_performance >= self.history[-1]

    def early_stopping_check(self):
        if self.is_patience_lost(self.early_stopping_patience):
            self.model.stop_training = True


    def reduce_lr_on_plateau(self):
        if self.is_patience_lost(self.plateau_patience):
            new_lr = float(K.get_value(self.model.optimizer.lr)) * self.reduction_rate
            K.set_value(self.model.optimizer.lr, new_lr)
            print(f"\n{'# ' *20}\nReduced learning rate to {new_lr}.\n{'# ' *20}\n")

    def model_checkpoint(self, bacc):
        if bacc > self.best_bacc:
            # remove previous checkpoints to save space

            self.best_bacc = bacc
            self.model.save('../models/best_model_' + str(self.fold) + '.h5')
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
        self.history.append(bacc)
        self.model_checkpoint(bacc)
        # self.reduce_lr_on_plateau()
        # self.early_stopping_check()
        logs['f1'] = score

        if epoch == self.swa_epoch:
            self.swa_weights = self.model.get_weights()

        if epoch > self.swa_epoch and bacc > self.best_bacc:
            self.swa_control +=1
            for i in range(len(self.swa_weights)):
                self.swa_weights[i] = (self.swa_weights[i] *
                                       self.swa_control + self.model.get_weights()[i]) / (self.swa_control + 1)

        else:
            pass

    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print('Final model parameters set to stochastic weight average.')
        self.model.save(self.swa_filepath)
        print('Final stochastic averaged weights saved to file.')


