import pickle
from keras.preprocessing import text, sequence
import pandas as pd
import argparse
import sys
from tqdm import tqdm
# tqdm.pandas()

def tokenize(X_train,X_test,max_features, maxlen):
    print("Tokenizando")
    tok = text.Tokenizer(num_words=max_features, lower=True, oov_token='outro')
    tok.fit_on_texts(list(X_train) + list(X_test))
    X_train = tok.texts_to_sequences(X_train)
    X_test = tok.texts_to_sequences(X_test)
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen, padding='post', value=0)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen, padding='post', value=0)
    # # saving
    # with open('../tokenizers/' + lang + '_tokenizer.pickle', 'wb') as handle:
    #     pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return tok, X_train, X_test

def save_multi_inputs(X_test_small,features, lang):
    with open('../tokenizers/' + lang + '_small_tokenizer.pickle', 'wb') as handle:
        pickle.dump(X_test_small, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../tokenizers/' + lang + '_features_tokenizer.pickle', 'wb') as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

# def generate_test_tokenizers(train,test,max_features, maxlen,type_model,languages):
#
#     for lang in languages:
#
#         train_new = train[train["language"] == lang]
#         test_new = test[test["language"] == lang]
#
#         train_new['title'] = train_new['title'].str.lower()
#         test_new['title'] = test_new['title'].str.lower()
#
#         train_new["title"] = train_new["title"].progress_apply(lambda x: clean_numbers(x))
#         train_new["title"] = train_new["title"].progress_apply(lambda x: replace_typical_misspell(x, lang))
#         train_new["title"] = train_new["title"].progress_apply(lambda x: clean_text(x))
#
#         test_new["title"] = test_new["title"].progress_apply(lambda x: clean_numbers(x))
#         test_new["title"] = test_new["title"].progress_apply(lambda x: replace_typical_misspell(x, lang))
#         test_new["title"] = test_new["title"].progress_apply(lambda x: clean_text(x))
#
#         if type_model == 'three':
#             train_new = build_features(train_new)
#             test_new = build_features(test_new)
#
#         X_train = train_new['title']
#         X_test = test_new["title"]
#
#         print("Tokenizando")
#         tok = text.Tokenizer(num_words=max_features, lower=True)
#         tok.fit_on_texts(list(X_train) + list(X_test))
#         X_test = tok.texts_to_sequences(X_test)
#         X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
#         # saving
#         with open('../tokenizers/' + lang + '_tokenizer.pickle', 'wb') as handle:
#             pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#         if type_model =='three':
#
#             X_test_small = test_new["small_title"]
#             X_test_small = tok.texts_to_sequences(X_test_small)
#             X_test_small = sequence.pad_sequences(X_test_small, maxlen=6)
#             X_test_features = test_new[
#                 ['n_words', 'length', 'n_capital_letters', 'n_numbers', 'small_length', 'small_n_capital_letters',
#                  'small_n_numbers']].values
#
#             save_multi_inputs(X_test_small, X_test_features, lang)

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Predict script')


    parser.add_argument('--model', help='Local of training', default='normal')

    return parser.parse_args(args)

# if __name__ == '__main__':
#     train = pd.read_csv("../../dados/train.csv")
#     test = pd.read_csv("../../dados/test.csv")
#
#     max_features = 100000
#     maxlen = 20
#
#     args = sys.argv[1:]
#     args = parse_args(args)
#
#     languages = ['portuguese', 'spanish']
#
#     generate_test_tokenizers(train,test,max_features,maxlen,args.model,languages)