import numpy as np
import config
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import KFold
import pandas as pd
from ast import literal_eval
from anago.preprocessing import IndexTransformer
from keras.models import load_model
from anago.layers import CRF

def evaluate():
    train = pd.read_csv(config.data_folder + "train.csv", converters={"pos": literal_eval, "tag": literal_eval})
    x_train = [x.split() for x in train['sentence'].tolist()]
    y_train = train['tag'].tolist()

    p = IndexTransformer(use_char=True)
    p.load('../models/best_transform.it')

    oof_data = []
    oof_data_pred = []

    skf = KFold(n_splits=config.nfolds, random_state=config.seed, shuffle=True)

    for n_fold, (train_indices, val_indices) in enumerate(skf.split(x_train)):
        x_val = list(np.array(x_train)[val_indices])
        y_val = list(np.array(y_train)[val_indices])

        lengths = map(len, x_val)
        x_val = p.transform(x_val)


        oof_data.extend([x for line in x_val for x in line])



        model = load_model('../models/best_model_' + str(n_fold) + '.h5', custom_objects={'CRF': CRF})
        y_pred = model.predict(x_val,
                               verbose=True)
        y_pred = p.inverse_transform(y_pred, lengths)
        oof_data_pred.extend([pred for line in y_pred for pred in line])

    bacc = balanced_accuracy_score(oof_data,oof_data_pred)
    print("Final CV: ", bacc*100)



