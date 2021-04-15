from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from math import sqrt
from LoadData import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from torch import nn
from torch.autograd import Variable

'''
dataset
数据集说明
'''

dataset = csv2datasets()
training_set = dataset.iloc[0:3500, 0:1].values  # 前(2426-300=2126)天的开盘价作为训练集,表格从0开始计数，2:3 是提取[2:3)列，前闭后开,故提取出C列开盘价
test_set = dataset.iloc[3500:, 0:1].values  # 后300天的开盘价作为测试集

# 归一化
sc = MinMaxScaler(feature_range=(0, 1))  # 定义归一化：归一化到(0，1)之间
training_set_scaled = sc.fit_transform(training_set)  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
test_set = sc.transform(test_set)  # 利用训练集的属性对测试集进行归一化
print(training_set_scaled)

x_train = []
y_train = []

x_test = []
y_test = []

# 测试集：csv表格中前2426-300=2126天数据
# 利用for循环，遍历整个训练集，提取训练集中连续60天的开盘价作为输入特征x_train，第61天的数据作为标签，for循环共构建2426-300-60=2066组数据。
for i in range(60, len(training_set_scaled)):
    x_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])
# 对训练集进行打乱
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
# 将训练集由list格式变为array格式
x_train, y_train = np.array(x_train), np.array(y_train)

# 使x_train符合RNN(PyTorch)输入要求：[循环核时间展开步数， 送入样本数， 每个时间步输入特征个数]。
# 此处整个数据集送入，送入样本数为x_train.shape[0]即2066组数据；输入60个开盘价，预测出第61天的开盘价，循环核时间展开步数为60; 每个时间步送入的特征是某一天的开盘价，只有1个数据，故每个时间步输入特征个数为1
x_train = np.reshape(x_train, (60, x_train.shape[0],  1))
# 测试集：csv表格中后300天数据
# 利用for循环，遍历整个测试集，提取测试集中连续60天的开盘价作为输入特征x_train，第61天的数据作为标签，for循环共构建300-60=240组数据。
for i in range(60, len(test_set)):
    x_test.append(test_set[i - 60:i, 0])
    y_test.append(test_set[i, 0])
# 测试集变array并reshape为符合RNN(PyTorch)输入要求：[循环核时间展开步数， 送入样本数， 每个时间步输入特征个数]
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (60, x_test.shape[0],  1))

x_train, y_train, x_test, y_test = torch.from_numpy(x_train).to(torch.float32), torch.from_numpy(y_train).to(torch.float32), torch.from_numpy(
    x_test).to(torch.float32), torch.from_numpy(y_test).to(torch.float32)

'''model'''


class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.gru1 = nn.GRU(input_size=1, hidden_size=64, num_layers=1, dropout=0.2, batch_first=True)
        self.linear = nn.Linear(in_features=64, out_features=1)

    def forward(self, _x):
        _x, _ = self.gru1(_x)
        print(_x)
        s, b, h = _x.shape
        _x = _x.contiguous().view(s * b, h)  # 转换成线性层的输入格式
        _x = self.linear(_x)
        _x = _x.view(s, b, -1)
        return _x


model = GRU().cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# 开始训练
for e in range(1000):
    var_x = Variable(x_train).cuda()
    var_y = Variable(y_train).cuda()
    # 前向传播
    out = model(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (e + 1) % 100 == 0:  # 每 100 次输出结果
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.data[0]))

# '''save'''
#
#
#
# '''predict'''
# # 测试集输入模型进行预测
# model = model.eval() # 转换成测试模式
#
# data_X = data_X.reshape(-1, 1, 2)
# data_X = torch.from_numpy(data_X)
# var_data = Variable(data_X)
# pred_test = model(var_data) # 测试集的预测结果
# # 改变输出的格式
# pred_test = pred_test.view(-1).data.numpy()
# # 对预测数据还原---从（0，1）反归一化到原始范围
# predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# # 对真实数据还原---从（0，1）反归一化到原始范围
# real_stock_price = sc.inverse_transform(test_set[60:])
# # 画出真实数据和预测数据的对比曲线
# plt.plot(real_stock_price, color='red', label='MaoTai Stock Price')
# plt.plot(predicted_stock_price, color='blue', label='Predicted MaoTai Stock Price')
# plt.title('40M Temperature Prediction')
# plt.xlabel('Time')
# plt.ylabel('Temperature')
# plt.legend()
# plt.show()
#
# '''evaluate'''
#
# # calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
# mse = mean_squared_error(predicted_stock_price, real_stock_price)
# # calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
# rmse = sqrt(mean_squared_error(predicted_stock_price, real_stock_price))
# # calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
# mae = mean_absolute_error(predicted_stock_price, real_stock_price)
# print('均方误差: %.6f' % mse)
# print('均方根误差: %.6f' % rmse)
# print('平均绝对误差: %.6f' % mae)
