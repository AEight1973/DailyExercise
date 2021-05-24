import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch import nn
from torch.autograd import Variable
import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import math

'''
New Dataset

数据集: weatherdata_2011_2021.xlsx
数据集包含： 崇明区近10年主要气象指标
I   具体指标：
    1)  日期: yyyy-mm-dd 星期X
    2)  最高气温: TT℃
    3)  最低气温: tt℃
    4)  天气: w
    5)  风向: d风 n级
II  中文转编码
    1)  天气： {'晴': 0, '多云': 1, '阴': 2,
               '小雨': 3, '阵雨': 4, '中雨': 5,
               '大雨': 6, '雷阵雨': 7, '暴雨': 8}
    2)  风向： {'北风': 0, '东北风': 1, '东风': 2, '东南风': 3,
               '南风': 4, '西南风': 5, '西风': 6, '西北风': 7}
'''

# 模拟周期数据
data = []
for i in range(2000):
    data.append(math.sin(i * 0.01))
data = np.array(data).reshape(-1, 1)
time = list(range(2000))

# 基本变量
batch_size = 256
n_feature = data.shape[1]
time_step = 150
epoch = 50
n_class = 1
learn_rate = 1e-2

train_set, test_set, train_time, test_time = train_test_split(data, time, train_size=0.7, shuffle=False)

# 归一化
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(train_set)
test_set_scaled = sc.transform(test_set)

x_train, y_train, x_test, y_test = [], [], [], []

# 将时间序列转化为数据集
for i in range(time_step, len(training_set_scaled)):
    x_train.append(training_set_scaled[i - time_step:i, :])
    y_train.append(training_set_scaled[i, :])
# 将训练集由list格式变为FloatTensor格式
x_train, y_train = torch.FloatTensor(x_train), torch.FloatTensor(y_train)
# 使x_train符合RNN(PyTorch)输入要求：[循环核时间展开步数， 送入样本数， 每个时间步输入特征个数]。
x_train = x_train.view(-1, time_step, n_feature)

for i in range(time_step, len(test_set_scaled)):
    x_test.append(test_set_scaled[i - time_step:i, :])
    y_test.append(test_set_scaled[i, :])
x_test, y_test = torch.FloatTensor(x_test), torch.FloatTensor(y_test)
x_test = x_test.view(-1, time_step, n_feature)

# 使用torch的TensorDataset进行数据集的分批处理
train_dl = [DataLoader(dataset=TensorDataset(x_train, y_train[:, i]), batch_size=batch_size)
            for i in range(n_feature)]
valid_dl = [DataLoader(dataset=TensorDataset(x_test, y_test[:, i]), batch_size=batch_size)
            for i in range(n_feature)]

'''
model

模型结构: 3 * GRU -> Linear -> 2 * GRU -> Linear
损失函数: 交叉熵损失函数 CrossEntropyLoss
优化器: Adam
'''


class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.gru1 = nn.GRU(input_size=n_feature, hidden_size=10, num_layers=3, dropout=0.2, batch_first=True)
        self.linear1 = nn.Linear(in_features=10, out_features=n_feature)
        self.gru2 = nn.GRU(input_size=n_feature, hidden_size=10, num_layers=2, dropout=0.2, batch_first=True)
        self.linear2 = nn.Linear(in_features=10, out_features=n_class)

    def forward(self, _x):
        _x, _ = self.gru1(_x)
        s, b, h = _x.shape
        _x = _x.contiguous().view(s * b, h)  # 转换成线性层的输入格式
        _x = self.linear1(_x)
        _x = _x.view(s, b, -1)
        _x, _ = self.gru2(_x)
        s, b, h = _x.shape
        _x = _x.contiguous().view(s * b, h)  # 转换成线性层的输入格式
        _x = self.linear2(_x)
        _x = _x.view(s, b, -1)
        return _x


# class GRU(nn.Module):
#     def __init__(self):
#         super(GRU, self).__init__()
#         self.gru1 = nn.LSTM(input_size=1, hidden_size=4, num_layers=2, dropout=0.2, batch_first=True)
#         self.linear = nn.Linear(in_features=4, out_features=1)
#
#     def forward(self, _x):
#         _x, _ = self.gru1(_x)
#         s, b, h = _x.shape
#         _x = _x.contiguous().view(s * b, h)  # 转换成线性层的输入格式
#         _x = self.linear(_x)
#         _x = _x.view(s, b, -1)
#         return _x


# 定义损失函数
criterion = nn.MSELoss()


Model = []
for i in range(n_feature):
    model = GRU().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    # 开始训练
    Loss, Valid_Loss = [], []
    for e in range(epoch):
        model.train()

        loss = 0
        for xb, yb in train_dl[i]:
            # 转化数据
            var_x = Variable(xb).cuda()
            var_y = Variable(yb).cuda()

            optimizer.zero_grad()

            # 前向传播
            out = model(var_x)

            _loss = criterion(out, var_y)

            # 反向传播
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()

            loss += _loss.item()

        # 加入验证集，评估模型
        model.eval()
        with torch.no_grad():
            valid_loss = sum(criterion(model(xb.cuda()), yb.cuda()) for xb, yb in valid_dl[i])

        print('--Epoch: {}, Loss: {:.5f}, Valid_Loss: {:.5f}'.format(e + 1, loss / len(train_dl[i]),
                                                                     valid_loss / len(valid_dl[i])))
        Loss.append(loss / len(train_dl))
        Valid_Loss.append(valid_loss / len(valid_dl))
    Model.append(model)

    # # 保存模型
    # path = 'model'
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # PATH = path + '/' + features[i] + '_' + datetime.datetime.now().strftime('%m%d%H%M')
    # torch.save(model, PATH)

'''predict'''
# 测试集输入模型进行预测
Model = [i.eval() for i in Model]  # 转换成测试模式

# 对真实数据进行绘图
data_maxtemp = data[:, [0]]
plt.plot(time, data_maxtemp, color='red', label='40M Real MaxTemperature')


# plt.show()

def predict(x, inverse=True):
    # 转化为Variable并训练
    var = Variable(x).cuda()
    pred = []
    for m in Model:
        _pred = m(var)
        # 改变输出的格式
        pred.append(_pred.data.cpu().numpy()[:, -1, 0].reshape(-1))
    pred = np.array(pred).T
    if inverse:
        # 对训练数据还原---从（0，1）反归一化到原始范围
        return sc.inverse_transform(pred)
    else:
        return pred


# 对训练数据进行绘图
train_data = predict(x_train)
# 画出真实数据和预测数据的对比曲线
train_maxtemp = train_data[:, [0]]
plt.plot(train_time[time_step:], train_maxtemp, color='blue', label='40M Train MaxTemperature')

# 对测试数据进行绘图
test_data = predict(x_test)
# 画出真实数据和预测数据的对比曲线
test_maxtemp = test_data[:, [0]]
plt.plot(test_time[time_step:], test_maxtemp, color='green', label='40M Test MaxTemperature')

# 对未来进行预测
# predict_time = []
# predict_data = list(sc.transform(data))
# _time = time[-1]
# for i in range(n_predict):
#     # if _time.day == 30:
#     #     _time = datetime.date(_time.year + 1, 6, 1)
#     # else:
#     #     _time += datetime.timedelta(days=1)
#     _time += datetime.timedelta(days=1)
#     x_predict = torch.FloatTensor(predict_data[-time_step:]).view(-1, time_step, n_feature)
#     predict_data += list(predict(x_predict, False))
#     predict_time.append(_time)
# predict_temp = sc.inverse_transform(np.array(predict_data[-n_predict:]))
# plt.plot(predict_time, predict_temp, color='orange', label='40M Predict Temperature')
#
# plt.title('40M Temperature Training Consult')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.show()
