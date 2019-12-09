import pandas as pd
import config
from anago.preprocessing import IndexTransformer
from anago.models import BiLSTMCRF
from ast import literal_eval
from anago.layers import CRF, crf_loss, crf_viterbi_accuracy
from keras_radam import RAdam
import numpy as np
from keras.models import load_model
import sys
import argparse
import os


def build_submission(y_pred, n_fold):
    sample = pd.read_csv(config.sample_submission)
    y_pred = [pred for line in y_pred for pred in line]
    sample['label'] = y_pred

    sample.to_csv('submission_' + str(n_fold) + '.csv', index=False)

def predict(model, p, x_test, n_fold):
    lengths = map(len, x_test)
    x_test = p.transform(x_test)
    y_pred = model.predict(x_test,
                           verbose=True)
    y_pred = p.inverse_transform(y_pred, lengths)

    build_submission(y_pred, n_fold)
    return y_pred

def load_and_predict():
    test = pd.read_csv(config.data_folder  + "test.csv", converters={"pos": literal_eval})
    x_test = [x.split() for x in test['sentence'].tolist()]

    p = IndexTransformer(use_char=True)
    p = p.load('../models/best_transform.it')

    model = BiLSTMCRF(char_vocab_size=p.char_vocab_size,
                      word_vocab_size=p.word_vocab_size,
                      num_labels=p.label_size,
                      word_embedding_dim=300,
                      char_embedding_dim=100,
                      word_lstm_size=100,
                      char_lstm_size=50,
                      fc_dim=100,
                      dropout=0.5,
                      embeddings=None,
                      use_char=True,
                      use_crf=True)

    model, loss = model.build()
    model.load_weights('../models/' + 'best_model.h5')

    predict(model, p, x_test)

def predict_with_folds(swa):
    test = pd.read_csv(config.data_folder + "test.csv", converters={"pos": literal_eval})
    x_test = [x.split() for x in test['sentence'].tolist()]

    p = IndexTransformer(use_char=True)
    p = p.load('../models/best_transform.it')
    lengths = map(len, x_test)
    x_test = p.transform(x_test)

    fold_result = []
    for n_fold in range(config.nfolds):

        path = '../models/best_model_' + str(n_fold)

        if swa:
            path += '_swa'

        model = load_model(path + '.h5',
                           custom_objects={'CRF': CRF,
                                           'RAdam': RAdam,
                                           'crf_loss': crf_loss,
                                           'crf_viterbi_accuracy': crf_viterbi_accuracy})
        y_pred = model.predict(x_test,
                               verbose=True)

        fold_result.append(y_pred)

    final_pred = np.mean(fold_result, axis=0)
    y_pred = p.inverse_transform(final_pred, lengths)
    build_submission(y_pred, 'fold')

def predict_with_emsemble(swa):
    test = pd.read_csv(config.data_folder + "test.csv", converters={"pos": literal_eval})
    x_test = [x.split() for x in test['sentence'].tolist()]

    p = IndexTransformer(use_char=True)
    p = p.load('../models/best_transform.it')
    lengths = map(len, x_test)
    x_test = p.transform(x_test)

    fold_result = []
    model_result=[]

    for folder_model in [x for x in os.listdir('../models/') if os.path.isdir(os.path.join('../models/',x))]:
        for n_fold in range(config.nfolds):

            path = '../models/' + str(folder_model) + '/best_model_' + str(n_fold)

            if swa:
                path += '_swa'

            model = load_model(path + '.h5',
                               custom_objects={'CRF': CRF,
                                               'RAdam': RAdam,
                                               'crf_loss': crf_loss,
                                               'crf_viterbi_accuracy': crf_viterbi_accuracy})
            y_pred = model.predict(x_test,
                                   verbose=True)

            fold_result.append(y_pred)

        final_pred = np.mean(fold_result, axis=0)
        model_result.append(final_pred)
        if folder_model == '9344':
            model_result.append(final_pred)

    model_result = np.mean(model_result, axis=0)
    y_pred = p.inverse_transform(model_result, lengths)
    build_submission(y_pred, 'emsemble')




def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Predict script')


    parser.add_argument("--swa", default=False, type=bool)
    parser.add_argument("--emsemble", default=False, type=bool)



    return parser.parse_args(args)

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)


    if args.emsemble:
        predict_with_emsemble(args.swa)
    else:
        predict_with_folds(args.swa)