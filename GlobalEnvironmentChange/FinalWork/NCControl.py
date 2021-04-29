# -*- coding:utf-8 -*-
import numpy as np
import netCDF4 as nc
import pandas as pd
import os


class NCFile:
    def __init__(self, _path, _refresh=1):
        self.path = _path
        self.refresh = _refresh
        if os.path.exists(_path):
            self.height, self.dataset = self.read_nc()
        else:
            self.dataset = self.write_nc()


    def write_nc(self):
        lon = np.linspace(119.885, 120.536, 652)
        lat = np.linspace(29.984, 29.358, 627)
        da = nc.Dataset(self.path, 'w', format='NETCDF4')
        da.createDimension('height', 627)  # 创建坐标点
        da.createDimension('feature', )
        da.createDimension('time', 652)  # 创建坐标点
        da.createVariable("lon", 'f', "lon")  # 添加coordinates 'f'为数据类型，不可或缺
        da.createVariable("lat", 'f', "lat")  # 添加coordinates 'f'为数据类型，不可或缺
        da.variables['lat'][:] = lat  # 填充数据
        da.variables['lon'][:] = lon  # 填充数据
        da.createVariable('u', 'f8', ('lat', 'lon'))  # 创建变量，shape=(627,652) 'f'为数据类型，不可或缺
        da.variables['u'][:] = data  # 填充数据
        da.close()

    def insert_nc(self):
        return 1

    def read_nc(self):
        return
