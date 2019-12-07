import numpy as np
import config
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import KFold
import pandas as pd
from ast import literal_eval
from anago.preprocessing import IndexTransformer
from keras.models import load_model
from anago.layers import CRF, crf_loss, crf_viterbi_accuracy
from keras_radam import RAdam
import sys
import argparse

def evaluate(swa):
    train = pd.read_csv(config.data_folder + "train.csv", converters={"pos": literal_eval, "tag": literal_eval})
    x_train = [x.split() for x in train['sentence'].tolist()]
    y_train = train['tag'].tolist()

    p = IndexTransformer(use_char=True)
    p = p.load('../models/best_transform.it')

    oof_data = []
    oof_data_pred = []

    skf = KFold(n_splits=config.nfolds, random_state=config.seed, shuffle=True)

    for n_fold, (train_indices, val_indices) in enumerate(skf.split(x_train)):

        x_val = list(np.array(x_train)[val_indices])
        y_val = list(np.array(y_train)[val_indices])
        print(y_val[:5])
        oof_data.extend([x for line in y_val for x in line])
        print(oof_data[:5])
        lengths = map(len, x_val)
        x_val = p.transform(x_val)

        path = '../models/best_model_' + str(n_fold)

        if swa:
            path += '_swa'

        model = load_model(path + '.h5',
                           custom_objects={'CRF': CRF,
                                           'RAdam': RAdam,
                                           'crf_loss' : crf_loss,
                                           'crf_viterbi_accuracy': crf_viterbi_accuracy})

        # model.load_weights('../models/best_model_' + str(n_fold) + '.h5')

        y_pred = model.predict(x_val,
                               verbose=True)
        print(y_pred[:5])
        y_pred = p.inverse_transform(y_pred, lengths)
        print(y_pred[:5])
        oof_data_pred.extend([pred for line in y_pred for pred in line])
        print(oof_data_pred[:5])

    bacc = balanced_accuracy_score(oof_data,oof_data_pred)
    print("Final CV: ", bacc*100)

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Predict script')


    parser.add_argument("--swa", default=False, type=bool)




    return parser.parse_args(args)

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)


    print("Evaluation")
    evaluate(args.swa)
