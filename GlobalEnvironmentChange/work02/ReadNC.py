import netCDF4
import pandas as pd
from netCDF4 import Dataset

nc_obj = Dataset('data/air_1948.mon.mean.nc')


def read():
    _lat = (nc_obj.variables['lat'][:])
    _lon = (nc_obj.variables['lon'][:])
    _time = (nc_obj.variables['time'][:])
    _air = (nc_obj.variables['air'][:])

    return _lat, _lon, _time, _air


if __name__ == '__name__':
    # 查看nc文件有些啥东东
    print(nc_obj)
    print('---------------------------------------')

    # 查看nc文件中的变量
    keys = list(nc_obj.variables.keys())
    print(keys)
    print('---------------------------------------')

    # 查看每个变量的信息
    for i in keys:
        print(nc_obj.variables[i])
    print('---------------------------------------')

    # 查看每个变量的属性
    for i in keys:
        print(nc_obj.variables[i].ncattrs())
        print(nc_obj.variables[i].units)
    print('---------------------------------------')

    # 读取数据值
    lat = (nc_obj.variables['lat'][:])
    lon = (nc_obj.variables['lon'][:])
    time = (nc_obj.variables['time'][:])
    air = (nc_obj.variables['air'][:])
    print(lat)
    print(lon)
    print(time)
    print(air)
