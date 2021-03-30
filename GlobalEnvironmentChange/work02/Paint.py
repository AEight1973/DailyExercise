import matplotlib.pyplot as plt
import numpy as np
import ReadNC
import datetime
import math

lat, lon, time, air = ReadNC.read()

air = np.array(air)


def show(_lat_code, _lon_code):
    x = list(time)
    y = list(air[:, _lat_code, _lon_code])

    aver = dict()
    for i in range(len(x)):
        _time = datetime.datetime(1800, 1, 1, 0, 0, 0) + datetime.timedelta(hours=x[i])
        if _time.year == 2020:
            break
        try:
            aver[_time.year].append(y[i])
        except:
            aver[_time.year] = [y[i]]
    X = list(aver.keys())
    Y = [np.mean(i) for i in list(aver.values())]
    plt.plot(X, Y)
    plt.show()
    print(math.cos(90 - lat_code * 2.5))
    Y1 = np.array(Y) * abs(math.cos(90 - lat_code * 2.5))
    plt.plot(X, Y1)
    plt.show()
    Y2 = np.convolve(Y, [0.2] * 5, mode='valid')
    plt.plot(X[2: -2], Y2)
    plt.show()



if __name__ == '__main__':
    lon = 0
    lat = -90
    lon_code = int(lon // 2.5)
    lat_code = int((90 - lat) // 2.5)
    show(lat_code, lon_code)
