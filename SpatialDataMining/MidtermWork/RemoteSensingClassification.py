from GDALTools import *
import numpy as np
import sklearn
import os


class RSImage:
    def __init__(self, _path):
        self.path = _path
        self.band = self._read()
        self.width, self.height = list(self.band.values())[0].shape

    def _read(self):
        _file = [i for i in os.listdir(self.path) if i[-3:] == 'TIF']
        _output = dict()
        for i in _file:
            info = i[: -4].split('_')[-1]
            if info[0] == 'B':
                _output['band' + info[1:]] = read_img(self.path + '/' + i)[2]
        return _output

    def band_composite(self, _red='4', _green='3', _blue='2'):
        _img = np.zeros((self.width, self.height, 3))
        _img[:, :, 0] = self.band['band' + _red]
        _img[:, :, 1] = self.band['band' + _green]
        _img[:, :, 2] = self.band['band' + _blue]
        return _img


def flez():
    return


if __name__ == '__main__':
    img = RSImage('data/LC08_L2SP_126058_20200824_20200905_02_T1')
