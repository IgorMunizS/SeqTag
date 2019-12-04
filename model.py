from keras.layers import LSTM, Dense, TimeDistributed, Embedding, Bidirectional, CuDNNLSTM,Dropout
from keras.models import Model, Input
from anago.layers import CRF
from keras.callbacks import ModelCheckpoint
import config
def get_model(max_len,max_features,embed_size,embedding_matrix,num_tags):
    sequence_input = Input(shape=(max_len,))

    embedding = Embedding(
        max_features,
        embed_size,
        weights=[embedding_matrix],
        trainable=config.train_embeddings,
        mask_zero=True)(sequence_input)

    x = Dropout(0.5)(embedding)
    x1 = Bidirectional(LSTM(100, return_sequences=True))(x)
    td = Dense(100, activation='tanh')(x1)
    crf = CRF(num_tags, sparse_target=False)  # CRF layer
    out = crf(td)  # output

    model = Model(sequence_input, out)

    return model, crf