import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.datasets import imdb

# keras实现LSTM网络
# Hyper parameters
from keras.utils import np_utils
from tensorflow.python.keras import Sequential

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
model.add(LSTM(nb_lstm_outputs, input_shape=input_shape))
model.add(Dense(nb_classes, activation='softmax', init=init_weights))  # 初始权重

model.summary()  # 打印模型
plot(model, to_file='lstm_model.png')  # 绘制模型结构图，并保存成图片

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])  # 编译模型
history = model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size, shuffle=True, verbose=1)  # 迭代训练

score = model.evaluate(X_test, Y_test, verbose=1)  # 模型评估
print('Test score:', score[0])
print('Test accuracy:', score[1])
