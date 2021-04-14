from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import GRU, Dropout, Dense
from numpy import concatenate, c_
from math import sqrt
from LoadData import *
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 配置TensorFlow
tf.compat.v1.disable_eager_execution()

config = tf.compat.v1.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True

session = tf.compat.v1.Session(config=config)

# 设置全局变量
batch_size = 64
epochs = 50
time_steps = 14


'''dataset'''

dataset = csv2datasets()
values = dataset.values
nb_classes = values.shape[1]
# 数据转化为 float32
values = values.astype('float32')
# 数据标准化 (0,1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# 时间序列转化为数据集
reframed = series_to_supervised(scaled, time_steps)
print('reframed data:', reframed.head())

# 分割成测试集与测试集
values = reframed.values
train_X, test_X, train_y, test_y = train_test_split(values[:, :-1], values[:, -1], test_size=0.2)
# reshape为符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
train_X = train_X.reshape((train_X.shape[0], time_steps, nb_classes))
test_X = test_X.reshape((test_X.shape[0], time_steps, nb_classes))
print('train_X.shape:', train_X.shape, 'train_y.shape:', train_y.shape, '\n')
print('test_X.shape:', test_X.shape, 'test_y.shape:', test_y.shape, '\n')

# 打乱训练集
np.random.seed(7)
np.random.shuffle(train_X)
np.random.seed(7)
np.random.shuffle(train_y)
tf.random.set_seed(7)


'''model'''

model = Sequential()
model.add(GRU(80, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(100))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error')

# 训练网络
history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y),
                    verbose=1, shuffle=False)

# 绘画loss和var_loss展示训练效果
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


'''predict'''

# 测试集输入模型进行预测
predicted_stock_price = model.predict(test_X)
# 对预测数据还原---从（0，1）反归一化到原始范围
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
# 对真实数据还原---从（0，1）反归一化到原始范围
real_stock_price = scaler.inverse_transform(scaled)
# 画出真实数据和预测数据的对比曲线
plt.plot(real_stock_price, color='red', label='MaoTai Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted MaoTai Stock Price')
plt.title('40M Temperature Prediction')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.show()


'''evaluate'''

# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(predicted_stock_price, real_stock_price)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = sqrt(mean_squared_error(predicted_stock_price, real_stock_price))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(predicted_stock_price, real_stock_price)
print('均方误差: %.6f' % mse)
print('均方根误差: %.6f' % rmse)
print('平均绝对误差: %.6f' % mae)
