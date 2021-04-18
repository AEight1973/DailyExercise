from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from torch.utils.data import TensorDataset, DataLoader
from LoadData import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from torch import nn
from torch.autograd import Variable
import datetime

'''
global
全局变量
'''

epoch = 50
time_step = 14
batch_size = 128
n_feature = 1

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

x_train = []
y_train = []

x_test = []
y_test = []

# 测试集：csv表格中前2426-300=2126天数据
# 利用for循环，遍历整个训练集，提取训练集中连续60天的开盘价作为输入特征x_train，第61天的数据作为标签，for循环共构建2426-300-60=2066组数据。
for i in range(time_step, len(training_set_scaled)):
    x_train.append(training_set_scaled[i - time_step:i, 0])
    y_train.append(training_set_scaled[i, 0])
# 将训练集由list格式变为array格式
x_train, y_train = torch.FloatTensor(x_train), torch.FloatTensor(y_train)

# 使x_train符合RNN(PyTorch)输入要求：[循环核时间展开步数， 送入样本数， 每个时间步输入特征个数]。
# 此处整个数据集送入，送入样本数为x_train.shape[0]即2066组数据；输入60个开盘价，预测出第61天的开盘价，循环核时间展开步数为60; 每个时间步送入的特征是某一天的开盘价，只有1个数据，故每个时间步输入特征个数为1
x_train = x_train.view(-1, time_step, n_feature)

# 测试集：csv表格中后300天数据
# 利用for循环，遍历整个测试集，提取测试集中连续60天的开盘价作为输入特征x_train，第61天的数据作为标签，for循环共构建300-60=240组数据。
for i in range(time_step, len(test_set)):
    x_test.append(test_set[i - time_step:i, 0])
    y_test.append(test_set[i, 0])
# 测试集变array并reshape为符合RNN(PyTorch)输入要求：[循环核时间展开步数， 送入样本数， 每个时间步输入特征个数]
x_test, y_test = torch.FloatTensor(x_test), torch.FloatTensor(y_test)
x_test = x_test.view(-1, time_step, n_feature)

# 使用torch的TensorDataset进行数据集的分批处理
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
valid_ds = TensorDataset(x_test, y_test)
valid_dl = DataLoader(dataset=valid_ds, batch_size=batch_size)

'''model'''


class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.gru1 = nn.GRU(input_size=1, hidden_size=10, num_layers=2, dropout=0.2, batch_first=True)
        self.linear = nn.Linear(in_features=10, out_features=1)

    def forward(self, _x):
        _x, _ = self.gru1(_x)
        s, b, h = _x.shape
        _x = _x.contiguous().view(s * b, h)  # 转换成线性层的输入格式
        _x = self.linear(_x)
        _x = _x.view(s, b, -1)
        return _x


model = GRU().cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# 开始训练
Loss, Valid_Loss = [], []
for e in range(epoch):
    model.train()

    loss = 0
    for xb, yb in train_dl:
        # 转化数据
        var_x = Variable(xb).cuda()
        var_y = Variable(xb).cuda()

        # 前向传播
        out = model(var_x)
        _loss = criterion(out, var_y)

        # 反向传播
        optimizer.zero_grad()
        _loss.backward()
        optimizer.step()

        loss += _loss.item()

    # 加入验证集
    model.eval()  # 评估模型
    with torch.no_grad():
        valid_loss = sum(criterion(model(xb.cuda()), yb.cuda()) for xb, yb in valid_dl)

    print('Epoch: {}, Loss: {:.5f}, Valid_Loss: {:.5f}'.format(e + 1, loss / len(train_dl), valid_loss / len(valid_dl)))
    Loss.append(loss / len(train_dl))
    Valid_Loss.append(valid_loss / len(valid_dl))

plt.plot(Loss, color='red', label='loss')
plt.plot(Valid_Loss, color='blue', label='valid_loss')
plt.title('Model Training')
plt.xlabel('epoch')
plt.ylabel('value')
plt.legend()
plt.show()

'''save'''

PATH = 'cache/model/temp_40m_single_{}'.format(datetime.datetime.now().strftime('%y%m%d%H%M%S'))
torch.save(model, PATH)

'''predict'''
# 测试集输入模型进行预测
model = model.eval()  # 转换成测试模式

# 对真实数据进行绘图
real_data = dataset.values
real_time = dataset.index
plt.plot(real_time, real_data, color='red', label='40M Real Temperature')

# 对训练数据进行绘图
var_train = Variable(x_train).cuda()
pred_train = model(var_train)
# 改变输出的格式
pred_train = pred_train.data.cpu().numpy()[:, -1, 0].reshape(-1, 1)
# 对训练数据还原---从（0，1）反归一化到原始范围
train_temp = sc.inverse_transform(pred_train)
# 画出真实数据和预测数据的对比曲线
train_time = real_time[time_step: 3500]
plt.plot(train_time, train_temp, color='blue', label='40M Train Temperature')

# 对测试数据进行绘图
var_test = Variable(x_test).cuda()
pred_test = model(var_test)
# 改变输出的格式
pred_test = pred_test.data.cpu().numpy()[:, -1, 0].reshape(-1, 1)
# 对测试数据还原---从（0，1）反归一化到原始范围
test_temp = sc.inverse_transform(pred_test)
# 画出真实数据和预测数据的对比曲线
test_time = real_time[3500 + time_step:]
plt.plot(test_time, test_temp, color='green', label='40M Test Temperature')

# 对未来一年进行预测
predict_time = []
predict_data = list(sc.transform(real_data).T[0])
_time = real_time[-1]
for i in range(367):
    if _time.hour == 0 and _time.month == 10:
        _time = datetime.datetime(_time.year + 1, 4, 1, 0, 0, 0)
    else:
        _time += datetime.timedelta(hours=12)
    var_predict = Variable(torch.FloatTensor(predict_data[-time_step:]).view(-1, time_step, n_feature)).cuda()
    pred_predict = model(var_predict)
    predict_data.append(pred_predict.data.cpu().numpy()[0, -1, 0])
    predict_time.append(_time)
predict_temp = sc.inverse_transform(np.array(predict_data[-367:]).reshape(-1, 1))
plt.plot(predict_time, predict_temp, color='orange', label='40M Predict Temperature')


plt.title('40M Temperature Training Consult')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.show()

'''evaluate'''

# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
real_temp = real_data[3500 + time_step:, 0:1]
mse = mean_squared_error(test_temp, real_temp)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = sqrt(mean_squared_error(test_temp, real_temp))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(test_temp, real_temp)
print('均方误差: %.6f' % mse)
print('均方根误差: %.6f' % rmse)
print('平均绝对误差: %.6f' % mae)
