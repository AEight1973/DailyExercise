import numpy as np

'''
空间插值
克里金空间插值法
'''


def space_kriging(train_x, train_y):
    import pyKriging as krige
    model = krige.kriging(train_x, train_y, name='simple')
    model.train(optimizer='ga')
    # model.predict([x, y])
    model.plot()


'''
时间插值
利用RNN中的GRU模型，对时间序列进行预测，对部分空缺值填充
'''


def time_gru(_dataset, _code, _station, epoch=50, time_step=6, cuda_or_not=True):
    from sklearn.preprocessing import MinMaxScaler
    from torch.utils.data import TensorDataset, DataLoader
    import torch
    from torch import nn
    from torch.autograd import Variable
    import datetime
    from sklearn.model_selection import train_test_split

    data = _dataset.values
    batch_size = 128
    n_feature = data.shape[1]
    _time = list(_dataset.index)

    train_set, test_set = train_test_split(data, train_size=0.8, shuffle=False)
    time_train, time_test = _time[:len(train_set)], _time[len(train_set):]

    # 归一化
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(train_set)
    test_set = sc.transform(test_set)

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    # 将时间序列转化为数据集
    for i in range(time_step, len(training_set_scaled)):
        if time_train[i - time_step] == time_train[i] - datetime.timedelta(hours=12) * time_step:
            x_train.append(training_set_scaled[i - time_step:i, :])
            y_train.append(training_set_scaled[i, :])
    # 将训练集由list格式变为array格式
    x_train, y_train = torch.FloatTensor(x_train), torch.FloatTensor(y_train)

    # 使x_train符合RNN(PyTorch)输入要求：[循环核时间展开步数， 送入样本数， 每个时间步输入特征个数]。
    x_train = x_train.view(-1, time_step, n_feature)
    y_train = y_train.view(-1, n_feature)

    for i in range(time_step, len(test_set)):
        if time_test[i - time_step] == time_test[i] - datetime.timedelta(hours=12) * time_step:
            x_test.append(test_set[i - time_step:i, :])
            y_test.append(test_set[i, :])
    x_test, y_test = torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    x_test = x_test.view(-1, time_step, n_feature)
    y_test = y_test.view(-1, n_feature)

    # 使用torch的TensorDataset进行数据集的分批处理
    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    valid_ds = TensorDataset(x_test, y_test)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=batch_size)

    '''model'''

    class GRU(nn.Module):
        def __init__(self):
            super(GRU, self).__init__()
            self.gru = nn.GRU(input_size=1, hidden_size=10, num_layers=2, dropout=0.2, batch_first=True)
            self.linear = nn.Linear(in_features=10, out_features=1)

        def forward(self, _x):
            _x, _ = self.gru(_x)
            s, b, h = _x.shape
            _x = _x.contiguous().view(s * b, h)  # 转换成线性层的输入格式
            _x = self.linear(_x)
            _x = _x.view(s, b, -1)
            return _x

    if cuda_or_not:
        model = GRU().cuda()
    else:
        model = GRU()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # 开始训练
    Loss, Valid_Loss = [], []
    for e in range(epoch):
        model.train()

        loss = 0
        for xb, yb in train_dl:
            # 转化数据
            if cuda_or_not:
                var_x = Variable(xb).cuda()
                var_y = Variable(xb).cuda()
            else:
                var_x = Variable(xb)
                var_y = Variable(xb)

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
            if cuda_or_not:
                valid_loss = sum(criterion(model(xb.cuda()), yb.cuda()) for xb, yb in valid_dl)
            else:
                valid_loss = sum(criterion(model(xb), yb) for xb, yb in valid_dl)

        print('Epoch: {}, Loss: {:.5f}, Valid_Loss: {:.5f}'.format(e + 1, loss / len(train_dl),
                                                                   valid_loss / len(valid_dl)))
        Loss.append(loss / len(train_dl))
        Valid_Loss.append(valid_loss / len(valid_dl))

    '''save'''

    _path = 'cache/data/' + _station + '/model'
    if not os.path.exists(_path):
        os.makedirs(_path)
    PATH = _path + '/interpolation_' + str(c)
    torch.save(model, PATH)

    '''predict'''
    model = model.eval()  # 转换成测试模式

    start = datetime.datetime(2008, 1, 1, 0)
    for i in range(time_step, len(_time)):
        if _time[i - time_step - 1] == _time[i] - datetime.timedelta(hours=12) * (time_step - 1):
            start = _time[i]
            break
    end = datetime.datetime(2019, 12, 31, 12)
    datelist = []
    while start <= end:
        datelist.append(start)
        start += datetime.timedelta(hours=12)

    predict_time = []
    predict_data = []
    for t in datelist:
        if t in _time:
            continue
        else:
            _data = []
            for i in range(time_step, 0, -1):
                _t = t - datetime.timedelta(hours=12) * i
                if _t in _time:
                    _data.append(list(_dataset.loc[_t]))
                else:
                    _data.append(predict_data[predict_time.index(_t)])
            var_predict = Variable(torch.FloatTensor(_data).view(-1, time_step, n_feature)).cuda()
            pred_predict = model(var_predict)
            predict_data.append(list(pred_predict.data.cpu().numpy()[0, -1, :]))
            predict_time.append(_time)
    return pd.DataFrame(predict_data, index=predict_time, columns=_dataset.columns)


if __name__ == '__main__':
    import pandas as pd
    import sounding_db as sd
    import json
    import os
    import datetime
    from tqdm import tqdm
    from RecordFailure import record_download

    # 提取中国探空站ID
    stationlist = pd.read_excel('UPAR_GLB_MUL_FTM_STATION.xlsx')
    for i in list(stationlist.loc[164: 252, '区站号']):
        print('开始插值站点' + str(i))
        with open('cache/data/' + str(i) + '/download.json') as f:
            download = json.load(f)
        available = [i for i in list(download.keys()) if download[i] == 0]
        feature = ['temperature', 'dewpoint', 'direction', 'speed']
        time = []
        dataset = dict()
        for j in tqdm(available):
            receive = sd.read(i, j)
            for l in receive:
                dataset[l['pressure']] += l[feature].values
            time.append(datetime.datetime(year=int(j[0:4]), month=int(j[4:6]), day=int(j[6:8]), hour=int(j[8:10])))
        predict = dict()
        for c, d in dataset.items():
            predict[c] = time_gru(pd.DataFrame(d, index=time, columns=feature), c, str(i))
        for time in predict[0].index:
            new_array = []
            for p in list(predict.keys()):
                new_array.append([p] + list(predict[p].loc[time]))
            path = 'cache/data/' + str(i)
            new_data = pd.DataFrame(new_array, columns=['pressure'] + feature)
            if os.path.exists(path):
                new_data.to_csv(path + '/' + str(i) + '_' + time.strftime('%Y%m%d%H') + '.csv')
                sd.csv2db(path + '/' + str(i) + '_' + time.strftime('%Y%m%d%H') + '.csv', None)
                record_download(str(i), time.strftime('%Y%m%d%H'), 'inter')
