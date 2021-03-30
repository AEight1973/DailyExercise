import datetime
import math
from ReadNC import read
import numpy as np
import matplotlib.pyplot as plt

Lat, Lon, Time, Air = read()
Air = np.array(Air)


def MovingAverage(_x, _y, _len):
    aver = dict()
    for i in range(len(_x)):
        _time = datetime.datetime(1800, 1, 1, 0, 0, 0) + datetime.timedelta(hours=_x[i])
        if _time.year == 2020:
            break
        try:
            aver[_time.year].append(_y[i])
        except:
            aver[_time.year] = [_y[i]]
    X = list(aver.keys())
    Y = [np.mean(i) for i in list(aver.values())]
    return X[2: -2], np.convolve(Y, [0.2] * 5, mode='valid')


def TempDiff(_x, _y):
    return {_y.index(max(_y)): max(_y), _y.index(min(_y)): min(_y)}


def TempRate(_x, _y):
    Y = [(_y[i + 9] - _y[i]) / 10 for i in range(len(_y) - 9)]
    return [_x[Y.index(max(Y))], max(Y)]


def TempWeight(_y, lat):
    return list(_y * abs(math.cos(lat * math.pi / 180)))


if __name__ == '__main__':
    time = Time[-20:]
    temp = np.zeros([len(Lat), len(Lon)])
    for _lon in Lon:
        code_lon = int(_lon // 2.5)
        for _lat in Lat:
            code_lat = int((90 - _lat) // 2.5)
            # Y = TempWeight(Air[-20:, code_lat, code_lon], _lat)
            Y = Air[-20:, code_lat, code_lon]
            temp[code_lat, code_lon] = TempRate(time, Y)[1]
    np.savetxt('EveryDecade.txt', temp)
    plt.imshow(temp)

    plt.show()

