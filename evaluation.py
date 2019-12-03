from utils.generator import DataGenerator
import numpy as np
import config
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

def evaluate(model,X_train,y_train, tags, label_encoder):

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=0.9, random_state=233)

    val_generator = DataGenerator(X_test[:10240], y_test[:10240], tags, batch_size=128, shuffle=False)

    y_pred = model.predict_generator(
        val_generator,
        workers=40,
        verbose=1
    )

    y_pred = np.argmax(y_pred, axis=-1)
    y_test = np.argmax(y_test, -1)

    y_pred = y_pred.flatten()
    y_test = y_test.flatten()

    y_pred = label_encoder.inverse_transform(y_pred)
    y_test = label_encoder.inverse_transform(y_test)

    y_pred = y_pred[y_pred != config.pad_seq_tag]
    y_test = y_test[y_test != config.pad_seq_tag]


    print("ACC-score is : {:.1%}".format(accuracy_score(y_test, y_pred)))
    print("BACC-score is : {:.1%}".format(balanced_accuracy_score(y_test, y_pred)))
    print("F1-score is : {:.1%}".format(f1_score(y_test, y_pred)))

