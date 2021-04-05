import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as k
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import load_model
import tensorflow as tf
from ImportData import read

'''TensorFlow基本配置'''
tf.compat.v1.disable_eager_execution()

config = tf.compat.v1.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True

session = tf.compat.v1.Session(config=config)

print('<---------- 成功载入TensorFlow ---------->')

'''数据集载入并生成训练集&验证机&测试集'''
images, label = read()

X_train, X_test, y_train, y_test = train_test_split(images, label, test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

print('<---------- 成功载入数据集 ---------->')

'''配置CNN模型'''
np.random.seed(1337)

# 全局变量
batch_size = 32
nb_classes = 6
epochs = 50
# input image dimensions
img_rows, img_cols = 100, 100
# 卷积滤波器的数量
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# 根据不同的backend定下不同的格式
if k.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_valid = X_valid.reshape(X_valid.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 类型转换
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_valid = X_valid.astype('float32')
X_train /= 255
X_test /= 255
X_valid /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(X_valid.shape[0], 'valid samples')

# 转换为one_hot类型
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
Y_valid = np_utils.to_categorical(y_valid, nb_classes)

print('<---------- 数据标准化成功 ---------->')

# 构建模型
model = Sequential()

model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                        padding='same',
                        input_shape=input_shape))  # 卷积层1
model.add(Activation('relu'))  # 激活层
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1])))  # 卷积层2
model.add(Activation('relu'))  # 激活层
model.add(MaxPooling2D(pool_size=pool_size))  # 池化层
model.add(Dropout(0.5))  # 神经元随机失活
model.add(Flatten())  # 拉成一维数据
model.add(Dense(128))  # 全连接层1
model.add(Activation('relu'))  # 激活层
model.add(Dropout(0.5))  # 随机失活
model.add(Dense(nb_classes))  # 全连接层2
model.add(Activation('softmax'))  # Softmax评分

# 编译模型
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('<---------- 开始训练模型 ---------->')

# 训练模型
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(X_valid, Y_valid))
model.save('weather_classify.h5')
model = load_model('weather_classify.h5')

print('<---------- 成功训练模型 ---------->')

# 评估模型
y_predict = model.predict(X_test)
y_pred = np.argmax(y_predict, axis=1)
print(y_pred)
print(classification_report(y_test, y_pred))
