import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import metpy.calc as mpcalc
from metpy.plots import Hodograph, SkewT
from metpy.units import units
import metpy
import numpy as np


def load_data(index):
    col_names = ['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed']
    df = pd.read_csv("58362_{}.csv", encoding='gbk'.format(index))
    df = df[col_names]
    df = df.dropna(axis=0, how='any')

    return df


def df2datasets(df):
    x = df['pressure', 'height', 'dewpoint', 'direction', 'speed'].values
    y = df['temperature'].values

    return x, y


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
