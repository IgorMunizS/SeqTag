import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import Adam, Nadam, SGD
from sklearn.model_selection import train_test_split
from keras_radam import RAdam
from ast import literal_eval
from utils.generator import DataGenerator
from model import get_model
from utils.tokenizer import tokenize, save_multi_inputs
from utils.embeddings import meta_embedding, CharVectorizer
# from utils.callbacks import Lookahead, CyclicLR
from sklearn.utils import class_weight
import argparse
import sys
import numpy as np
from utils.preprocessing import clean_numbers, clean_text
# from utils.features import build_features
from tqdm import tqdm
tqdm.pandas()
from keras.preprocessing import sequence
from sklearn.feature_extraction.text import HashingVectorizer
import config
from evaluation import evaluate
from anago.utils import load_data_and_labels, load_glove
from anago.models import BiLSTMCRF
from anago.preprocessing import IndexTransformer
from anago.trainer import Trainer
from utils.callbacks import BACCscore
from anago.utils import NERSequence
from anago.callbacks import F1score
from anago.utils import filter_embeddings
from utils.utils import f1_keras
from anago.layers import crf_viterbi_accuracy

def training(train,test):
    x_train = [x.split() for x in train['sentence'].tolist()]
    y_train = train['tag'].tolist()

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=233)


    print('Transforming datasets...')
    p = IndexTransformer(use_char=True)
    p.fit(x_train, y_train)

    embeddings = load_glove(config.glove_file)
    embeddings = filter_embeddings(embeddings, p._word_vocab.vocab, config.glove_size)

    model = BiLSTMCRF(char_vocab_size=p.char_vocab_size,
                      word_vocab_size=p.word_vocab_size,
                      num_labels=p.label_size,
                      word_embedding_dim=300,
                      char_embedding_dim=100,
                      word_lstm_size=100,
                      char_lstm_size=50,
                      fc_dim=100,
                      dropout=0.5,
                      embeddings=embeddings,
                      use_char=True,
                      use_crf=True)

    opt = Adam(lr=0.001)
    model, loss = model.build()
    model.compile(loss=loss, optimizer=opt, metrics=[crf_viterbi_accuracy])

    filepath = '../models/' + 'best_model'
    ckp = ModelCheckpoint(filepath + '.h5', monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True, mode='max',
                          save_weights_only=True)

    es = EarlyStopping(monitor='val_crf_viterbi_accuracy', min_delta=0.00001, patience=3, verbose=1, mode='max')
    rlr = ReduceLROnPlateau(monitor='val_crf_viterbi_accuracy', factor=0.2, patience=2, verbose=1, mode='max', min_delta=0.0001)

    callbacks = [ckp, es, rlr]

    train_seq = NERSequence(x_train, y_train, config.batch_size, p.transform)


    if x_val and y_val:
        valid_seq = NERSequence(x_val, y_val, config.batch_size, p.transform)
        f1 = F1score(valid_seq, preprocessor=p)
        callbacks.append(f1)

    model.fit_generator(generator=train_seq,
                        validation_data=valid_seq,
                              epochs=config.nepochs,
                              callbacks=callbacks,
                              verbose=True,
                              shuffle=True,
                              use_multiprocessing=True,
                              workers=42)



def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Predict script')


    parser.add_argument('--model', help='Local of training', default='normal')
    parser.add_argument("--cpu", default=False, type=bool)




    return parser.parse_args(args)

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)
    import os

    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    original_train = pd.read_csv(config.original_data_folder + 'train.conll', sep=' ', names=['Word', 'PoS', 'Tag'])
    train = pd.read_csv(config.data_folder + "train.csv", converters={"pos": literal_eval, "tag": literal_eval})
    test = pd.read_csv(config.data_folder  + "test.csv", converters={"pos": literal_eval})

    training(train,test)
