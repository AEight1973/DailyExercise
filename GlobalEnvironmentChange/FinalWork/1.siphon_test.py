import datetime
from metpy.units import units
from siphon.simplewebservice.wyoming import WyomingUpperAir

start = datetime.datetime(2010, 7, 29, 12)
end = datetime.datetime(2010, 7, 29, 12)

datelist = []
while start <= end:
    datelist.append(start)
    start += datetime.timedelta(hours=12)

# 选择下载站点（以上海宝山站为例）
stationlist = ['58362']

# 批量下载
for station in stationlist:
    for date in datelist:
        try:
            df = WyomingUpperAir.request_data(date, station)
            df.to_csv('data/' + station + '_' + date.strftime('%Y%m%d%H') + '.csv', index=False)
            print(date.strftime('%Y%m%d_%H') + '下载成功')
        except:
            print(date.strftime('%Y%m%d_%H') + '下载失败')
            pass
