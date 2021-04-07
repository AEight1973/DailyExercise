import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import metpy.calc as mpcalc
from metpy.plots import Hodograph, SkewT
from metpy.units import units
import metpy
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_data(index):
    col_names = ['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed', 'u_wind', 'v_wind']
    df = pd.read_csv("58362_{}.csv", encoding='gbk'.format(index))
    df = df[col_names]
    df = df.dropna(axis=0, how='any')

    return df


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def csv2datasets(height=4):
    import os
    from datetime import datetime
    file_list = os.listdir('data/')
    output = pd.DataFrame(columns=['pressure', 'temperature', 'dewpoint', 'direction', 'speed', 'u_wind', 'v_wind'])
    for filename in file_list:
        data = pd.read_csv('data/' + filename)
        print(data)
        print(data[data['height'] == height])
        data = data[data['height'] == height][['pressure', 'temperature', 'dewpoint', 'direction', 'speed', 'u_wind', 'v_wind']].values[0]
        print(output)
        time = filename.split('.')[0].split('_')[1]
        time = datetime(int(time[0:4]), int(time[4:6]), int(time[6:8]), int(time[8:10]))
        output.loc[time] = data

    return output


def paint(df):
    p = df['pressure'].values * units.hPa  # 单位：hPa
    t = df['temperature'].values * units.degC  # 单位：℃
    td = df['dewpoint'].values * units.degC  # 单位：℃
    wind_speed = df['speed'].values * units.knots  # 单位：knot
    wind_dir = df['direction'].values * units.degrees  # 单位：°
    u, v = mpcalc.wind_components(wind_speed, wind_dir)  # 计算水平风速u和v
    prof = mpcalc.parcel_profile(p, t[0], td[0]).to('degC')

    fig = plt.figure(figsize=(9, 9))
    skew = SkewT(fig)

    skew.plot(p, t, 'r', linewidth=2)  # 绘制层结曲线（红色）
    skew.plot(p, td, 'g', linewidth=2)  # 绘制露点曲线（绿色）
    skew.plot_barbs(p, u, v)  # 绘制风羽
    skew.plot(p, prof, 'k')

    skew.ax.set_ylim(1000, 200)
    skew.ax.set_xlim(-20, 60)

    # 显示图像
    plt.show()


if __name__ == '__main__':
    # load dataset
    dataset = csv2datasets()
    values = dataset.values
    # integer encode direction
    values[:, 4] = np_utils.to_categorical(values[:, 4], 8)
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # drop columns we don't want to predict
    # reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
    print(reframed)
