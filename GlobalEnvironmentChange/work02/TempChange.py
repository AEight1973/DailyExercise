import datetime
import math
from ReadNC import read
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import LinearRegress
import time as t
import pandas as pd

Lat, Lon, Time, Air, CO2 = read()
Air = np.array(Air)


def Resize(_x, _y):
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
    Y1 = [np.mean(i) for i in list(aver.values())]
    Y2 = [np.max(i) for i in list(aver.values())]
    Y3 = [np.min(i) for i in list(aver.values())]
    return X, Y1, Y2, Y3


def Detrend(_y):
    _x = list(range(len(_y)))
    [k, b] = LinearRegress.run(_x, _y)
    newlist = []
    for i in _x:
        newlist.append(_y[i] - (k[0] * i + b))
    return newlist


def MovingAverage(_y, _len):
    return np.convolve(_y, [0.2] * _len, mode='valid')


def TempDiff(_x, _y):
    return {_y.index(max(_y)): max(_y), _y.index(min(_y)): min(_y)}


def TempRate(_x, _y):
    Y = [(_y[i + 9] - _y[i]) / 10 for i in range(len(_y) - 9)]
    return [_x[Y.index(max(Y))], max(Y)]


def TempWeight(_y, lat):
    return list(_y * abs(math.cos(lat * math.pi / 180)))


if __name__ == '__main__':
    import LinearRegress

    temp1 = np.zeros([len(Lat), len(Lon)])
    temp2 = np.zeros([len(Lat), len(Lon)])
    temp3 = np.zeros([len(Lat), len(Lon)])
    temp4 = np.zeros([len(Lat), len(Lon)])
    temp5 = np.zeros([len(Lat), len(Lon)])
    temp6 = np.zeros([len(Lat), len(Lon)])
    temp7 = np.zeros([len(Lat), len(Lon)])

    for _lon in tqdm.tqdm(Lon):
        code_lon = int(_lon // 2.5)
        for _lat in Lat:
            code_lat = int((90 - _lat) // 2.5)
            time, Y1, Y2, Y3 = Resize(Time[:], Air[:, code_lat, code_lon])
            # temp1[code_lat, code_lon] = TempRate(time[-20:], Y[-20:])[1]  # 近20年年均温差
            temp2[code_lat, code_lon] = LinearRegress.run(time[-20:], Y1[-20:])[0]  # 近20年一元线性回归
            temp3[code_lat, code_lon] = LinearRegress.run(time[-20:], Y2[-20:])[0]  # 近20年一元线性回归
            temp4[code_lat, code_lon] = LinearRegress.run(time[-20:], Y3[-20:])[0]  # 近20年一元线性回归
            # temp3[code_lat, code_lon] = LinearRegress.run(CO2[-22: -2], MovingAverage(Y, 9)[-20:])[0]  # 平滑处理后的近20年一元线性回归
            # temp4[code_lat, code_lon] = np.corrcoef(CO2, Y)[0, 1]  # CO2与温度相关系数
            # temp5[code_lat, code_lon] = np.corrcoef(Detrend(CO2), Detrend(Y1))[0, 1]  # 平滑处理后的CO2与温度相关系数
            # temp6[code_lat, code_lon] = np.corrcoef(Detrend(CO2), Detrend(Y2))[0, 1]
            # temp7[code_lat, code_lon] = np.corrcoef(Detrend(CO2), Detrend(Y3))[0, 1]

    '''保存并可视化'''
    # np.savetxt('EveryDecade.txt', temp1)
    # np.savetxt('LinearTrend.txt', temp2)
    # np.savetxt('MovingAverage.txt', temp3)
    # np.savetxt('MovingAverage2.txt', temp4)
    # plt.imshow(temp1)
    # plt.show()
    plt.imshow(temp2)
    plt.show()
    plt.imshow(temp3)
    plt.show()
    plt.imshow(temp4)
    plt.show()
    # plt.imshow(temp5)
    # plt.show()
    # plt.imshow(temp6)
    # plt.show()
    # plt.imshow(temp7)
    # plt.show()
