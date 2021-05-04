import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import metpy.calc as mpcalc
from metpy.plots import Hodograph, SkewT
from metpy.units import units
import metpy

col_names = ['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed']
df = pd.read_csv("cache/data/58362/58362_2009040100.csv", encoding='gbk')
df = df[col_names]
df = df.dropna(axis=0, how='any')

# 将数据转化为pint.Quantity类型
p = df['pressure'].values * units.hPa  # 单位：hPa
T = df['temperature'].values * units.degC  # 单位：℃
Td = df['dewpoint'].values * units.degC  # 单位：℃
wind_speed = df['speed'].values * units.knots  # 单位：knot
wind_dir = df['direction'].values * units.degrees  # 单位：°

# 计算cape_cin
u, v = mpcalc.wind_components(wind_speed, wind_dir)  # 计算水平风速u和v
prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
cape, cin = mpcalc.cape_cin(pressure=p, temperature=T, dewpoint=Td, parcel_profile=prof)

fig = plt.figure(figsize=(9, 9))
skew = SkewT(fig)

# skew.plot(p, T, 'r', linewidth=2)  # 绘制层结曲线（红色）
# skew.plot(p, Td, 'g', linewidth=2)  # 绘制露点曲线（绿色）
# skew.plot_barbs(p, u, v)  # 绘制风羽
# skew.plot(p, prof, 'k')

# skew.ax.set_ylim(1000, 200)
# skew.ax.set_xlim(-20, 60)

# 显示图像
# plt.show()
