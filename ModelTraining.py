from matplotlib import pyplot
import os
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.models import load_model
import time
from math import sqrt
from sklearn.metrics import mean_squared_error
import numpy as np
from Feature import Feature


# split the data into features and label
def split_features_and_y(data, time_stamp, features_number):
    split_data = []
    for i in range(time_stamp+1):
        if i == 0:
            split_data = data[:, :((i + 1)*features_number)-2]
        else:
            split_data = np.hstack([split_data, data[:, i*features_number:((i + 1)*features_number)-2]])
    y_label = data[:, -2:]
    return split_data, y_label


# create the model
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


# get the features from the csv file
def get_extracted_features():
    i = 1
    features_data = []
    dirpath = os.path.dirname(os.path.realpath(__file__)) + "/resource/features"
    print("getting feature....")
    for filename in os.listdir(dirpath)[:50]:
        dataset = read_csv("resource/features/" + filename, header=0)
        values = dataset.values
        data = Feature.series_to_supervised(values, timesteps, 1)
        if i == 1:
            features_data = data
        else:
            features_data = np.vstack([features_data, data])
        print(i, '/', 1802)
        i += 1
    return features_data


timesteps = 3
# number of total features including 2 of the labels, arousal and valance
n_features = 148
# get all the features from csv file
features_data = get_extracted_features()
# separate the data into train, validation and test
train = features_data[:int(len(features_data)/2), :]
val = features_data[int(len(features_data)/2):int(len(features_data)/1.2), :]
test = features_data[int(len(features_data)/1.2):, :]
# split the data into features and labels
train_X, train_y = split_features_and_y(train, timesteps, n_features)
val_X, val_y = split_features_and_y(val, timesteps, n_features)
test_X, test_y = split_features_and_y(test, timesteps, n_features)
# reshape the data that is suitable for the model
train_X = train_X.reshape(train_X.shape[0], timesteps+1, n_features-2)
val_X = val_X.reshape(val_X.shape[0], timesteps+1, n_features-2)
test_X = test_X.reshape(test_X.shape[0], timesteps+1, n_features-2)
# create the model
model = get_model()
# train the model
history = model.fit(train_X, train_y, epochs=500, batch_size=50, validation_data=(val_X, val_y), verbose=2, shuffle=False)
print("saving model..")
# save the model
model.save(os.path.dirname(os.path.realpath(__file__)) + "/resource/model/LSTM.h5")
# display the training progress
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# make the prediction for calculating RMSE
print("start predicting...")
prediction = model.predict(test_X)
print("calculating RMSE...")
rmse = sqrt(mean_squared_error(prediction, test_y))
print("RMSE: ", rmse)

pyplot.show()