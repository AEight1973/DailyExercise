import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.datasets import imdb

# keras实现LSTM网络
# Hyper parameters
from keras.utils import np_utils
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dense

batch_size = 128
nb_epoch = 10
nb_time_steps = 12
dim_input_vector = 1
nb_classes = 10


# 载入数据lmdb
(X_train, Y_train), (X_test, Y_test) = imdb.load_data()

X_train = X_train.reshape(-1, 28, 28) / 255
X_test = X_test.reshape(-1, 28, 28) / 255

Y_train = np_utils.to_categorical(Y_train, num_classes=10)
Y_test = np_utils.to_categorical(Y_test, num_classes=10)

input_shape = (nb_time_steps, dim_input_vector)
# nb_time_steps时间步 ， dim_input_vector 表示输入向量的维度，也等于n_features，列数
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

Y_train = np_utils.to_categorical(y_train, nb_classes)  # 转为类别向量，nb_classes为类别数目
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Build LSTM network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
inv_y = scaler.inverse_transform(test_X)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

