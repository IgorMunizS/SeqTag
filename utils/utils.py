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