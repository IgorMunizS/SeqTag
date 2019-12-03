from keras.layers import LSTM, Dense, TimeDistributed, Embedding, Bidirectional, CuDNNLSTM,SpatialDropout1D
from keras.models import Model, Input
from keras_contrib.layers import CRF
from keras.callbacks import ModelCheckpoint

def get_model(max_len,max_features,embed_size,embedding_matrix,num_tags):
    sequence_input = Input(shape=(max_len,))

    embedding = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False, mask_zero=False)(sequence_input)
    x = SpatialDropout1D(0.3)(embedding)
    x1 = Bidirectional(LSTM(256, return_sequences=True))(x)
    x2 = Bidirectional(LSTM(128, return_sequences=True))(x1)
    td = Dense(50, activation="relu")(x2)
    crf = CRF(num_tags)  # CRF layer
    out = crf(td)  # output

    model = Model(sequence_input, out)

    return model, crf