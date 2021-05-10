from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch import nn
from torch.autograd import Variable
import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

'''Dataset
数据集: 崇明历史气候记录.txt
数据集包含： 崇明区近10年六月主要气象指标
I   具体指标：
    1)  日期: yyyy-mm-dd 星期X
    2)  最高气温: TT℃
    3)  最低气温: tt℃
    4)  天气: w1(~w2)
    5)  风向: d1(~d2) n1(-n2)[~n3(-n4)]
II  中文转编码
    1)  天气： {'晴': 0, '多云': 1, '阴': 2,
               '小雨': 3, '阵雨': 4, '中雨': 5,
               '大雨': 6, '雷阵雨': 7, '暴雨': 8}
    2)  风向： {'北风': 0, '东北风': 1, '东风': 2, '东南风': 3,
               '南风': 4, '西南风': 5, '西风': 6, '西北风': 7}
'''

# 读取文件
with open('data/崇明历史气候记录.txt', 'r+', encoding='utf8') as f:
    file = f.read().split('\n')
    _data = np.array(file).reshape((-1, 5))

# 中文转码字典
weather2dict = {'晴': 0, '多云': 1, '阴': 2,
               '小雨': 3, '阵雨': 4, '中雨': 5,
               '大雨': 6, '雷阵雨': 7, '暴雨': 8}
wind2dict = {'北风': 0, '东北风': 1, '东风': 2, '东南风': 3,
             '南风': 4, '西南风': 5, '西风': 6, '西北风': 7}

# 清洗数据
data = []
for i in _data[1:, 1:]:
    # 最高 & 最低气温
    t1 = int(i[0][:-1])
    t2 = int(i[1][:-1])
    # 天气
    if '~' in i[2]:
        W = i[2].split('~')
    elif '转' in i[2]:
        W = i[2].split('转')
    else:
        W = [i[2]]
    if '雨' in W[-1]:
        w = W[-1][-2:]
    elif '雨' in W[0]:
        w = W[0][-2:]
    else:
        w = W[0]
    # 风向 & 风速
    if i[3] == '暂无实况':
        d = '东南风'
        v = 2
    else:
        wind = i[3].split(' ')
        if '~' in wind[0]:
            d = wind[0].split('~')[0]
        elif '转' in wind[0]:
            d = wind[0].split('转')[0]
        else:
            d = wind[0]
        V = wind[1]
        if V == '微风':
            V = '1级'
        elif '风' in V:
            V = V.split('风')[-1]
        if '~' in V:
            V = V.split('~')[0]
        elif '转' in V:
            V = V.split('转')[0]
        if '-' in V:
            v = (int(V.split('-')[0]) + int(V.split('-')[1][0])) / 2
        elif V == '小于3级':
            v = 2
        else:
            v = int(V[0])
    data.append([t1, t2, weather2dict[w], wind2dict[d], v])

data = np.array(data)
time = [datetime.date(int(i[:4]), int(i[5:7]), int(i[8:10])) for i in _data[1:, 0].reshape(-1)]
dataset = pd.DataFrame(data, columns=_data[0, 1:].reshape(-1), index=time)

# 基本变量
batch_size = 128
n_feature = data.shape[1]
time_step = 7

train_set, test_set = train_test_split(data, train_size=0.8, shuffle=False)

# 归一化
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(train_set)
test_set_scaled = sc.transform(test_set)

x_train, y_train, x_test, y_test = [], [], [], []

# 将时间序列转化为数据集
for i in range(time_step, len(training_set_scaled)):
    x_train.append(training_set_scaled[i - time_step:i, :])
    y_train.append(training_set_scaled[i, :])
# 将训练集由list格式变为array格式
x_train, y_train = torch.FloatTensor(x_train), torch.FloatTensor(y_train)
# 使x_train符合RNN(PyTorch)输入要求：[循环核时间展开步数， 送入样本数， 每个时间步输入特征个数]。
x_train = x_train.view(-1, time_step, n_feature)
y_train = y_train.view(-1, n_feature)

for i in range(time_step, len(test_set_scaled)):
    x_test.append(test_set_scaled[i - time_step:i, :])
    y_test.append(test_set_scaled[i, :])
x_test, y_test = torch.FloatTensor(x_test), torch.FloatTensor(y_test)
x_test = x_test.view(-1, time_step, n_feature)
y_test = y_test.view(-1, n_feature)

# 使用torch的TensorDataset进行数据集的分批处理
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
valid_ds = TensorDataset(x_test, y_test)
valid_dl = DataLoader(dataset=valid_ds, batch_size=batch_size)

# '''model'''
#
# class GRU(nn.Module):
#     def __init__(self):
#         super(GRU, self).__init__()
#         self.gru = nn.GRU(input_size=1, hidden_size=10, num_layers=2, dropout=0.2, batch_first=True)
#         self.linear = nn.Linear(in_features=10, out_features=1)
#
#     def forward(self, _x):
#         _x, _ = self.gru(_x)
#         s, b, h = _x.shape
#         _x = _x.contiguous().view(s * b, h)  # 转换成线性层的输入格式
#         _x = self.linear(_x)
#         _x = _x.view(s, b, -1)
#         return _x
#
# if cuda_or_not:
#     model = GRU().cuda()
# else:
#     model = GRU()
#
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
#
# # 开始训练
# Loss, Valid_Loss = [], []
# for e in range(epoch):
#     model.train()
#
#     loss = 0
#     for xb, yb in train_dl:
#         # 转化数据
#         if cuda_or_not:
#             var_x = Variable(xb).cuda()
#             var_y = Variable(xb).cuda()
#         else:
#             var_x = Variable(xb)
#             var_y = Variable(xb)
#
#         # 前向传播
#         out = model(var_x)
#         _loss = criterion(out, var_y)
#
#         # 反向传播
#         optimizer.zero_grad()
#         _loss.backward()
#         optimizer.step()
#
#         loss += _loss.item()
#
#     # 加入验证集
#     model.eval()  # 评估模型
#     with torch.no_grad():
#         if cuda_or_not:
#             valid_loss = sum(criterion(model(xb.cuda()), yb.cuda()) for xb, yb in valid_dl)
#         else:
#             valid_loss = sum(criterion(model(xb), yb) for xb, yb in valid_dl)
#
#     print('Epoch: {}, Loss: {:.5f}, Valid_Loss: {:.5f}'.format(e + 1, loss / len(train_dl),
#                                                                valid_loss / len(valid_dl)))
#     Loss.append(loss / len(train_dl))
#     Valid_Loss.append(valid_loss / len(valid_dl))
#
# '''save'''
#
# _path = 'cache/data/' + _station + '/model'
# if not os.path.exists(_path):
#     os.makedirs(_path)
# PATH = _path + '/interpolation_' + str(c)
# torch.save(model, PATH)
#
# '''predict'''
# model = model.eval()  # 转换成测试模式
#
# start = datetime.datetime(2008, 1, 1, 0)
# for i in range(time_step, len(_time)):
#     if _time[i - time_step - 1] == _time[i] - datetime.timedelta(hours=12) * (time_step - 1):
#         start = _time[i]
#         break
# end = datetime.datetime(2019, 12, 31, 12)
# datelist = []
# while start <= end:
#     datelist.append(start)
#     start += datetime.timedelta(hours=12)
#
# predict_time = []
# predict_data = []
# for t in datelist:
#     if t in _time:
#         continue
#     else:
#         _data = []
#         for i in range(time_step, 0, -1):
#             _t = t - datetime.timedelta(hours=12) * i
#             if _t in _time:
#                 _data.append(list(_dataset.loc[_t]))
#             else:
#                 _data.append(predict_data[predict_time.index(_t)])
#         var_predict = Variable(torch.FloatTensor(_data).view(-1, time_step, n_feature)).cuda()
#         pred_predict = model(var_predict)
#         predict_data.append(list(pred_predict.data.cpu().numpy()[0, -1, :]))
#         predict_time.append(_time)
# return pd.DataFrame(predict_data, index=predict_time, columns=_dataset.columns)