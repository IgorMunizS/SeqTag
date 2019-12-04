import pandas as pd
import config
from anago.preprocessing import IndexTransformer
from anago.models import BiLSTMCRF
from ast import literal_eval


def build_submission(y_pred):
    sample = pd.read_csv(config.sample_submission)
    y_pred = [pred for line in y_pred for pred in line]
    sample['label'] = y_pred

    sample.to_csv('submission.csv', index=False)

def predict(model, p, x_test):
    lengths = map(len, x_test)
    x_test = p.transform(x_test)
    y_pred = model.predict(x_test)
    y_pred = p.inverse_transform(y_pred, lengths)

    print(y_pred)
    build_submission(y_pred)
    return y_pred

def load_and_predict():
    test = pd.read_csv(config.data_folder  + "test.csv", converters={"pos": literal_eval})
    x_test = [x.split() for x in test['sentence'].tolist()]

    p = IndexTransformer(use_char=True)
    p.load('../models/best_transform.it')

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

if __name__ == '__main__':
   load_and_predict()