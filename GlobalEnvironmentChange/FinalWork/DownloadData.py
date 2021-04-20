import datetime
from siphon.simplewebservice.wyoming import WyomingUpperAir
import os
import pandas as pd

start = datetime.datetime(2008, 1, 1, 0)
end = datetime.datetime(2020, 12, 31, 12)

datelist = []
while start <= end:
    datelist.append(start)
    start += datetime.timedelta(hours=12)

# 选择下载站点（以上海宝山站 ['58362'] 为例）
stationfile = pd.read_excel('UPAR_GLB_MUL_FTM_STATION.xlsx')
stationlist = list(stationfile['区站号'])

# 设置下载区间 (数据集共计817个站点数据 其中中国站点为 [164: 252])
# 设置区间起始点 单点下载时间较长 每次下载20个站点
station_range_start = 100
station_range_step = 20

# 批量下载
for station in stationlist[station_range_start: station_range_start + station_range_step]:
    station = str(station)
    for date in datelist:
        try:
            # 新建文件夹
            dirs = 'data/' + station
            if not os.path.exists(dirs):
                os.makedirs(dirs)

            # 如文件已存在，则已下载
            filepath = dirs + '/' + station + '_' + date.strftime('%Y%m%d%H') + '.csv'
            if os.path.exists(filepath):
                print(date.strftime('%Y%m%d_%H') + '已下载')
                continue
            else:
                df = WyomingUpperAir.request_data(date, station)
                df.to_csv(filepath, index=False)
            print(date.strftime('%Y%m%d_%H') + '下载成功')
        except:
            print(date.strftime('%Y%m%d_%H') + '下载失败')
            pass
