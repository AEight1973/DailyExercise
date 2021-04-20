import datetime
from siphon.simplewebservice.wyoming import WyomingUpperAir
import os
import pandas as pd

start = datetime.datetime(2009, 4, 1, 0)
end = datetime.datetime(2019, 9, 30, 12)

datelist = []
while start <= end:
    datelist.append(start)
    start += datetime.timedelta(hours=12)

# 选择下载站点（以上海宝山站为例）
stationfile = pd.read_excel('UPAR_CHN_MUL_STATION.xlsx')
stationlist = list(stationfile['区站号'])

# 批量下载
for station in stationlist:
    station = str(station)
    for date in datelist:
        try:
            df = WyomingUpperAir.request_data(date, station)
            dirs = 'data/' + station
            filepath = dirs + '/' + station + '_' + date.strftime('%Y%m%d%H') + '.csv'
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            if not os.path.exists(filepath):
                df.to_csv(filepath, index=False)
            print(date.strftime('%Y%m%d_%H') + '下载成功')
        except:
            print(date.strftime('%Y%m%d_%H') + '下载失败')
            pass
