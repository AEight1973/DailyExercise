import matplotlib.pyplot as plt
import numpy as np
import ReadNC
import datetime

lat, lon, time, air = ReadNC.read()

air = np.array(air)
x = list(time)
y = list(air[:, 0, 0])

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
# plt.plot(X, Y)
# plt.show()

import LinearRegress
LinearRegress.run(X, Y)
