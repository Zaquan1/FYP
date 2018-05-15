from matplotlib import pyplot
import os, errno
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from math import sqrt
from sklearn.metrics import mean_squared_error
import numpy as np
import miscellaneous as misc


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
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mean_squared_error', metrics=['mse'], optimizer='adam')
    return model


# get the features from the csv file
def get_extracted_features():
    i = 1
    features_data = []
    dirpath = os.path.dirname(os.path.realpath(__file__)) + "/resource/features"
    print("getting feature....")
    for filename in os.listdir(dirpath)[:20]:
        dataset = read_csv("resource/features/" + filename, header=0)
        values = dataset.values
        data = misc.series_to_supervised(values, timesteps, 1)
        if i == 1:
            features_data = data
        else:
            features_data = np.vstack([features_data, data])
        print(i, '/', 1802)
        i += 1
    return features_data


# create dir for storing model
try:
    os.mkdir('resource')
    os.mkdir('resource/model')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

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

# create the model for arousal
model_arousal = get_model()
print(train_y.shape)
# train the model for arousal
historyArousal = model_arousal.fit(train_X, train_y[:, 0], epochs=200, batch_size=100, validation_data=(val_X, val_y[:, 0]), verbose=2, shuffle=False)
print("saving model..")
# save the model
model_arousal.save(os.path.dirname(os.path.realpath(__file__)) + "/resource/model/LSTMArousal.h5")
# display the training progress
pyplot.plot(historyArousal.history['loss'], label='train_arousal')
pyplot.plot(historyArousal.history['val_loss'], label='test_arousal')
pyplot.legend()

# create the model for valance
model_valance = get_model()
# train model for valance
historyValance = model_valance.fit(train_X, train_y[:, 1], epochs=200, batch_size=100, validation_data=(val_X, val_y[:, 1]), verbose=2, shuffle=False)
print("saving model..")
# save the model
model_valance.save(os.path.dirname(os.path.realpath(__file__)) + "/resource/model/LSTMValance.h5")
# display the training progress
pyplot.plot(historyValance.history['loss'], label='train_valance')
pyplot.plot(historyValance.history['val_loss'], label='test_valance')
pyplot.legend()

# make the prediction for calculating RMSE
prediction_arousal = model_arousal.predict(test_X)
prediction_valance = model_valance.predict(test_X)
print("calculating RMSE...")
rmse_arousal = sqrt(mean_squared_error(prediction_arousal, test_y[:, 0]))
rmse_valance = sqrt(mean_squared_error(prediction_valance, test_y[:, 1]))
print("RMSE arousal: ", rmse_arousal, "RMSE valance: ", rmse_valance)

pyplot.show()