import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import Adam, Nadam
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
from utils.utils import pad_tokens
from sklearn.feature_extraction.text import HashingVectorizer
import config
from evaluation import evaluate

def __training(X_train,y_train,max_features,maxlen,embedding_matrix,embed_size,tags):

    model, crf = get_model(maxlen,max_features,embed_size,embedding_matrix,len(tags))

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.9, random_state=233)


    train_generator = DataGenerator(X_train, y_train, tags, batch_size=config.batch_size)

    val_generator = DataGenerator(X_val, y_train, tags, batch_size=config.batch_size, shuffle=False)


    opt = Adam(lr=0.001)

    model.compile(loss=crf.loss_function, optimizer=opt, metrics=[crf.accuracy])


    filepath = '../models/' + 'best_model'
    ckp = ModelCheckpoint(filepath + '.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min',
                          save_weights_only=True)

    es = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5, verbose=1, mode='min')
    # rlr = ReduceLROnPlateau(monitor='val_f1-score', factor=0.2, patience=3, verbose=1, mode='max', min_delta=0.0001)
    # swa = SWA('../models/best_' + str(smmodel) + '_' + str(backbone) + '_' + str(n_fold) + '_swa.h5', 10)

    # clr = CyclicLR(base_lr=0.0003, max_lr=0.001,
    #                step_size=35000, reduce_on_plateau=1, monitor='val_loss', reduce_factor=10)

    callbacks_list = [ckp, es]

    print("Treinando")

    model.fit_generator(generator=train_generator,
                        validation_data=val_generator,
                        callbacks=callbacks_list,
                        epochs=config.nepochs,
                        use_multiprocessing=True,
                        workers=42)

    return model

def training(train,test,original_train):


    train['sentence'] = train['sentence'].str.lower()
    test['sentence'] = test['sentence'].str.lower()

    # stopwords = RemoveStopWords(lang)

    train.drop([52495, 89263], inplace=True) # sentences with more than 150 tokens

    train["sentence"] = train["sentence"].progress_apply(lambda x: clean_numbers(x))
    train["sentence"] = train["sentence"].progress_apply(lambda x: clean_text(x))

    test["sentence"] = test["sentence"].progress_apply(lambda x: clean_numbers(x))
    test["sentence"] = test["sentence"].progress_apply(lambda x: clean_text(x))

    # Check if tokens are safe (total number has to be 701106)
    a = []
    b = [x.split() for x in test['sentence'].tolist()]
    for x in b:
        a.extend(x)
    print(len(a))

    tags = original_train['Tag'].unique()
    tags = np.append(tags,config.pad_seq_tag)
    print(tags)
    label_encoder = LabelEncoder().fit(tags)

    X_train = train["sentence"]
    X_test = test["sentence"]

    max_features = 300000
    maxlen = 150

    tok, X_train, X_test = tokenize(X_train, X_test, max_features, maxlen)
    max_features = min(max_features, len(tok.word_index) + 1)

    y_train = train['tag'].tolist()
    y_train = pad_tokens(y_train, maxlen, label_encoder)

    # Generate char embedding without preprocess
    text = (train['sentence'].tolist() + test["sentence"].tolist())
    char_vectorizer = CharVectorizer(max_features,text)
    char_embed_size = char_vectorizer.embed_size
    #
    glove_embedding_matrix = meta_embedding(tok,config.glove_file,max_features,config.glove_size)
    # fast_embedding_matrix = meta_embedding(tok, max_features, embed_size)
    #
    char_embedding = char_vectorizer.get_char_embedding(tok)
    #
    #
    embedding_matrix = np.concatenate((glove_embedding_matrix, char_embedding), axis=1)
    embed_size = config.glove_size + config.char_size

    model = __training(X_train,y_train,max_features,maxlen,embedding_matrix,embed_size,tags)

    evaluate(model, X_train, y_train, tags, label_encoder)


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Predict script')


    parser.add_argument('--model', help='Local of training', default='normal')




    return parser.parse_args(args)

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)

    original_train = pd.read_csv(config.original_data_folder + 'train.conll', sep=' ', names=['Word', 'PoS', 'Tag'])
    train = pd.read_csv(config.data_folder + "train.csv", converters={"pos": literal_eval, "tag": literal_eval})
    test = pd.read_csv(config.data_folder  + "test.csv", converters={"pos": literal_eval})

    training(train,test,original_train)
