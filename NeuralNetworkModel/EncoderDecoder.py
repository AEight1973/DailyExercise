from torch import nn
import torch.nn.functional as F
from ConvRNN import ConvGRU, ConvLSTM
from CNN import ConvEncoder, ConvDecoder


class EncoderDecoderGRU(nn.Module):
    def __init__(self, hidden_dim, b_size, lstm_dims):
        super(EncoderDecoderGRU, self).__init__()
        self.cnn_encoder = ConvEncoder(hidden_dim)
        self.cnn_decoder = ConvDecoder(b_size=b_size, inp_dim=hidden_dim)

        self.lstm_encoder = ConvGRU(
            input_size=(16, 16),
            input_dim=hidden_dim,
            hidden_dim=lstm_dims,
            kernel_size=(3, 3),
            num_layers=3,
            batchnorm=False,
            batch_first=True,
            activation=F.tanh
        )

        self.lstm_decoder = ConvGRU(
            input_size=(16, 16),
            input_dim=hidden_dim,
            hidden_dim=lstm_dims,
            kernel_size=(3, 3),
            num_layers=3,
            batchnorm=False,
            batch_first=True,
            activation=F.tanh
        )


class EncoderDecoderLSTM(nn.Module):
    def __init__(self, hidden_dim, b_size, lstm_dims):
        super(EncoderDecoderLSTM, self).__init__()
        self.cnn_encoder = ConvEncoder(hidden_dim)
        self.cnn_decoder = ConvDecoder(b_size=b_size, inp_dim=hidden_dim)

        self.lstm_encoder = ConvLSTM(
            input_size=(16, 16),
            input_dim=hidden_dim,
            hidden_dim=lstm_dims,
            kernel_size=(3, 3),
            num_layers=3,
            batchnorm=False,
            batch_first=True,
            activation=F.tanh
        )

        self.lstm_decoder = ConvLSTM(
            input_size=(16, 16),
            input_dim=hidden_dim,
            hidden_dim=lstm_dims,
            kernel_size=(3, 3),
            num_layers=3,
            batchnorm=False,
            batch_first=True,
            activation=F.tanh
        )
