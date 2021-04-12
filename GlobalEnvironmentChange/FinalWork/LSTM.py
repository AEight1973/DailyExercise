from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from numpy import concatenate, c_
from math import sqrt
from LoadData import *
import tensorflow as tf

# keras实现LSTM网络
# Hyper parameters

# 配置TensorFlow
tf.compat.v1.disable_eager_execution()

config = tf.compat.v1.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True

session = tf.compat.v1.Session(config=config)

# 设置全局变量
batch_size = 72
epochs = 10
time_steps = 363

# load dataset
dataset = csv2datasets()
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
# 主成分分析 PCA
values = c_[pca(values[:, [0, 2, 3, 4, 5, 6]]), values[:, 1]]
nb_classes = values.shape[1]
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, time_steps)
# drop columns we don't want to predict
drop_list = list(range(- nb_classes, -1))
reframed.drop(reframed.columns[drop_list], axis=1, inplace=True)
print(reframed.head())

# split into train and test sets
values = reframed.values
# n_train_hours = 3000
# train = values[:n_train_hours, :]
# test = values[n_train_hours:, :]
# split into input and outputs
# train_X, train_y = train[:, :-1], train[:, -1]
# test_X, test_y = test[:, :-1], test[:, -1]
train_X, test_X, train_y, test_y = train_test_split(values[:, :-1], values[:, -1], test_size=0.2)
# train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.2)
# reshape input to be 3D [samples, timesteps, features]
print(train_X.shape)
train_X = train_X.reshape((train_X.shape[0], time_steps, nb_classes))
# valid_X = valid_X.reshape((valid_X.shape[0], time_steps, valid_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], time_steps, nb_classes))
print('train_X.shape:', train_X.shape, 'train_y.shape:', train_y.shape, '\n')
# print('valid_X.shape:', valid_X.shape, 'valid_y.shape:', valid_y.shape, '\n')
print('test_X.shape:', test_X.shape, 'test_y.shape:', test_y.shape, '\n')

model = Sequential()
model.add(LSTM(80, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y),
                    verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# # make a prediction
# yhat = model.predict(test_X)
# test_x = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# # invert scaling for forecast
# inv_yhat = concatenate((yhat, test_x[:, 1:]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:, 0]
# # invert scaling for actual
# inv_y = scaler.inverse_transform(test_x)
# inv_y = inv_y[:, 0]
# # calculate RMSE
# rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
# print('Test RMSE: %.3f' % rmse)

# plot prediction
plt.figure(figsize=(24, 8))
train_predict = model.predict(train_X)
# valid_predict = model.predict(valid_X)
test_predict = model.predict(test_X)
plt.plot(values[:, -1], c='b')
plt.plot([x for x in train_predict], c='g')
plt.plot([None for _ in train_predict] + [x for x in test_predict], c='y')
# plt.plot([None for _ in train_predict] + [None for _ in valid_predict] + [x for x in test_predict], c='r')
plt.show()
