from matplotlib import pyplot
import os
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
import time
from math import sqrt
from sklearn.metrics import mean_squared_error
import numpy as np


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        agg = concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg


def split_features_and_y(data, time_stamp, features_number):
    split_data = []
    for i in range(time_stamp+1):
        if i == 0:
            split_data = data[:, :((i + 1)*features_number)-2]
        else:
            split_data = np.hstack([split_data, data[:, i*features_number:((i + 1)*features_number)-2]])
    y_label = data[:, -2:]
    return split_data, y_label


def get_model():
    model = Sequential()
    model.add(LSTM(512, input_shape=(4, 146),
                   return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(256))
    model.add(Dropout(0.1))
    model.add(Dense(256))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='tanh'))
    model.compile(loss='mean_squared_error', metrics=['mse'], optimizer='adam')
    return model


def get_extracted_features():
    i = 1
    features_data = []
    dirpath = os.path.dirname(os.path.realpath(__file__)) + "/resource/features"
    print("getting feature....")
    for filename in os.listdir(dirpath):
        dataset = read_csv("resource/features/" + filename, header=0)
        values = dataset.values
        data = series_to_supervised(values, timesteps, 1)
        if i == 1:
            features_data = data
        else:
            features_data = np.vstack([features_data, data])
        print(i, '/', 1802)
        i += 1
    return features_data


timesteps = 3
# n_features including 2 of the labels, arousal and valance
n_features = 148
start = time.time()
features_data = get_extracted_features()
end = time.time()
print("time: ", end-start)
train = features_data[:int(len(features_data)/2), :]
val = features_data[int(len(features_data)/2):int(len(features_data)/1.2), :]
test = features_data[int(len(features_data)/1.2):, :]

train_X, train_y = split_features_and_y(train, timesteps, n_features)
val_X, val_y = split_features_and_y(val, timesteps, n_features)
test_X, test_y = split_features_and_y(test, timesteps, n_features)
print("train: ", train_X.shape, train_y.shape)
print("val: ", val_X.shape, val_y.shape)
print("test: ", test_X.shape, test_y.shape)

train_X = train_X.reshape(train_X.shape[0], timesteps+1, n_features-2)
val_X = val_X.reshape(val_X.shape[0], timesteps+1, n_features-2)
test_X = test_X.reshape(test_X.shape[0], timesteps+1, n_features-2)
print("trainRe: ", train_X.shape)
print("valRe: ", val_X.shape)
print("test_x:", test_X.shape)

model = get_model()
history = model.fit(train_X, train_y, epochs=500, batch_size=50, validation_data=(val_X, val_y), verbose=2, shuffle=False)
print("saving model..")
model.save(os.path.dirname(os.path.realpath(__file__)) + "/resource/model/LSTM.h5")
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
print("start predicting...")
prediction = model.predict(test_X)
print("calculating RMSE...")
rmse = sqrt(mean_squared_error(prediction, test_y))
print("RMSE: ", rmse)

pyplot.show()
