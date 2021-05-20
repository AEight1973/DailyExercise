from torch import nn


class GRUSimple(nn.Module):
    def __init__(self, n_feature, n_hidden, n_class):
        super(GRUSimple, self).__init__()
        self.gru1 = nn.GRU(input_size=n_feature, hidden_size=n_hidden, num_layers=3, dropout=0.2, batch_first=True)
        self.linear1 = nn.Linear(in_features=n_hidden, out_features=n_feature)
        self.gru2 = nn.GRU(input_size=n_feature, hidden_size=n_hidden, num_layers=2, dropout=0.2, batch_first=True)
        self.linear2 = nn.Linear(in_features=n_hidden, out_features=n_class)

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


class LSTMSimple(nn.Module):
    def __init__(self, n_feature, n_hidden, n_class):
        super(LSTMSimple, self).__init__()
        self.lstm1 = nn.LSTM(input_size=n_feature, hidden_size=n_hidden, num_layers=3, dropout=0.2, batch_first=True)
        self.linear1 = nn.Linear(in_features=n_hidden, out_features=n_feature)
        self.lstm2 = nn.LSTM(input_size=n_feature, hidden_size=n_hidden, num_layers=2, dropout=0.2, batch_first=True)
        self.linear2 = nn.Linear(in_features=n_hidden, out_features=n_class)

    def forward(self, _x):
        _x, _ = self.lstm1(_x)
        s, b, h = _x.shape
        _x = _x.contiguous().view(s * b, h)  # 转换成线性层的输入格式
        _x = self.linear1(_x)
        _x = _x.view(s, b, -1)
        _x, _ = self.lstm2(_x)
        s, b, h = _x.shape
        _x = _x.contiguous().view(s * b, h)  # 转换成线性层的输入格式
        _x = self.linear2(_x)
        _x = _x.view(s, b, -1)
        return _x
