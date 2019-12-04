from typing import Optional, List
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer
import config
import numpy as np
import tensorflow as tf


def meta_embedding(tok,embedding_file,max_features,embed_size,lang='portuguese'):
    print("Generating Embedding")
    snowball_stemmer = SnowballStemmer(lang)

    embeddings_index = {}
    with open(embedding_file, encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    word_index = tok.word_index
    word_index[config.pad_seq_tag] = 0
    # word_index[config.unk_token] = 999999

    # prepare embedding matrix
    num_words = min(max_features, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, embed_size))
    # unknown_vector = np.random.normal(size=embed_size)
    unknown_vector = np.zeros((embed_size,), dtype=np.float32) - 1.
    pad_vector = np.zeros((embed_size,), dtype=np.float32) - 2.

    # print(unknown_vector[:5])
    for key, i in word_index.items():
        if i >= max_features:
            continue
        if i == 0: #pad
            embedding_matrix[i] = pad_vector
            continue

        word = key
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = key.lower()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = key.upper()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = key.capitalize()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = snowball_stemmer.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue

        embedding_matrix[i] = unknown_vector
    return embedding_matrix

class CharVectorizer:

    def __init__(self, max_features,text):

        self.text = text
        self.max_features = max_features

        self.vectorizer = CountVectorizer(analyzer='char', binary=False, decode_error='strict',
                                     encoding='utf-8', input='content',
                                     lowercase=True, max_df=1.0, max_features=100, min_df=1,
                                     ngram_range=(1, 1), preprocessor=None, stop_words=None,
                                     strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
                                     tokenizer=None, vocabulary=None)
        self.vectorizer.fit(self.text)
        self.embed_size = len(self.vectorizer.vocabulary_)

    def get_char_embedding(self,tok):

        word_index = tok.word_index
        # prepare embedding matrix
        num_words = min(self.max_features, len(word_index) + 1)
        embedding_matrix = np.zeros((num_words, self.embed_size))
        for key, i in word_index.items():
            if i >= self.max_features:
                continue
            word = key
            embedding_vector = self.vectorizer.transform([str(word)]).toarray().astype(np.float32)
            embedding_matrix[i] = embedding_vector

        return embedding_matrix