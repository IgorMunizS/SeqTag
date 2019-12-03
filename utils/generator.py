import numpy as np

import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, X, Y=None, classes=None, maxlen=150, batch_size=32,
                 shuffle=True):
        'Initialization'

        self.X = X
        self.n_inputs = len(self.X)


        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.classes = classes
        self.n_classes = len(self.classes)
        self.maxlen = maxlen

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        X_temp = [self.X[k] for k in indexes]

        Y_temp = [self.Y[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(X_temp, Y_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'


        self.indexes = np.arange(len(self.X))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, x, Y):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        y = np.empty((self.batch_size, self.maxlen, self.n_classes), dtype=np.float)
        for i in range(len(Y)):
            for j in range(self.maxlen):

                y[i, j, :] = keras.utils.to_categorical(Y[i][j], num_classes=self.n_classes)



        X = np.array(x)

        return X, y