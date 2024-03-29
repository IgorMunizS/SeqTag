import config
import numpy as np
import keras.backend as K

def pad_tokens(y_train, maxlen, label_encoder):
    padded_y_train = np.zeros((len(y_train), maxlen), dtype=np.int)
    for i,sample in enumerate(y_train):
        pad = [config.pad_seq_tag]*(maxlen - len(sample))
        sample.extend(pad)
        padded_y_train[i] = label_encoder.transform(sample)
    return padded_y_train

def per_class_accuracy(y_preds,y_true):
    class_labels = [0,1,2,3,4,5,6,7,8] # 9 is PAD
    return [K.mean([
        (y_true[pred_idx] == K.round(y_pred)) for pred_idx, y_pred in enumerate(y_preds)
      if y_true[pred_idx] == int(class_label)
                    ]) for class_label in class_labels]

def inverse_transform(self, y, label_encoder, lengths=None):
    """Return label strings.

    Args:
        y: label id matrix.
        lengths: sentences length.

    Returns:
        list: list of list of strings.
    """
    y = np.argmax(y, -1)
    inverse_y = [label_encoder.inverse_transform(ids) for ids in y]
    if lengths is not None:
        inverse_y = [iy[:l] for iy, l in zip(inverse_y, lengths)]

    return inverse_y