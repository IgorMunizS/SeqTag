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
from anago.utils import load_data_and_labels, load_glove
from anago.models import BiLSTMCRF, ELModel
from anago.preprocessing import IndexTransformer, ELMoTransformer
from anago.trainer import Trainer
from utils.callbacks import BACCscore
from anago.utils import NERSequence
from anago.callbacks import F1score
from anago.utils import filter_embeddings
from anago.layers import crf_viterbi_accuracy
from predict import predict
from sklearn.model_selection import KFold

### BEST CONFIG SO FAR ####
#Fold 1: 93.37
#Fold Pred: 93.651


def training(train,test, fold):
    x_train = [x.split() for x in train['sentence'].tolist()]
    y_train = train['tag'].tolist()

    x_test = [x.split() for x in test['sentence'].tolist()]

    print('Transforming datasets...')
    p = IndexTransformer(use_char=True)
    p.fit(x_train + x_test, y_train)

    skf = KFold(n_splits=config.nfolds, random_state=config.seed, shuffle=True)

    embeddings = load_glove(config.glove_file)
    embeddings_fast = load_glove(config.glove_file)
    embeddings_wang = load_glove(config.wang_file)

    embeddings = filter_embeddings(embeddings, p._word_vocab.vocab, config.glove_size)
    embeddings_fast = filter_embeddings(embeddings_fast, p._word_vocab.vocab, config.fasttext_size)
    embeddings_wang = filter_embeddings(embeddings_wang, p._word_vocab.vocab, config.wang_size)

    embeddings = np.concatenate((embeddings, embeddings_fast, embeddings_wang), axis=1)

    for n_fold, (train_indices, val_indices) in enumerate(skf.split(x_train)):

        if n_fold >= fold:
            print("Training fold: ", n_fold)
            x_val = list(np.array(x_train)[val_indices])
            y_val = list(np.array(y_train)[val_indices])

            x_train_spl = list(np.array(x_train)[train_indices])
            y_train_spl = list(np.array(y_train)[train_indices])


            model = BiLSTMCRF(char_vocab_size=p.char_vocab_size,
                              word_vocab_size=p.word_vocab_size,
                              num_labels=p.label_size,
                              word_embedding_dim=1800,
                              char_embedding_dim=50,
                              word_lstm_size=300,
                              char_lstm_size=300,
                              fc_dim=100,
                              dropout=0.3,
                              embeddings=embeddings,
                              use_char=True,
                              use_crf=True)

            opt = Adam(lr=0.001)
            model, loss = model.build()
            model.compile(loss=loss, optimizer=opt, metrics=[crf_viterbi_accuracy])


            es = EarlyStopping(monitor='val_crf_viterbi_accuracy',
                               patience=3,
                               verbose=1,
                               mode='max',
                               restore_best_weights=True)

            rlr = ReduceLROnPlateau(monitor='val_crf_viterbi_accuracy',
                                    factor=0.2,
                                    patience=2,
                                    verbose=1,
                                    mode='max')

            callbacks = [es,rlr]

            train_seq = NERSequence(x_train_spl, y_train_spl, config.batch_size, p.transform)


            if x_val and y_val:
                valid_seq = NERSequence(x_val, y_val, config.batch_size, p.transform)
                f1 = F1score(valid_seq, preprocessor=p, fold=n_fold)
                callbacks.append(f1)

            model.fit_generator(generator=train_seq,
                                validation_data=valid_seq,
                                epochs=config.nepochs,
                                callbacks=callbacks,
                                verbose=True,
                                shuffle=True,
                                use_multiprocessing=True,
                                workers=12)



            p.save('../models/best_transform.it')
            model.load_weights('../models/best_model_' + str(n_fold) + '.h5')
            predict(model, p , x_test, n_fold)

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Predict script')


    parser.add_argument('--fold', help='specific fold to train', default=0, type=int)
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

    print("TREINANDO")
    training(train,test, args.fold)
