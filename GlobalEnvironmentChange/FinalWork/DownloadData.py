import datetime
from siphon.simplewebservice.wyoming import WyomingUpperAir
import os
import pandas as pd
from tqdm import tqdm
from RecordFailure import record_download
import json
from time import sleep


def download():
    start = datetime.datetime(2008, 1, 1, 0)
    end = datetime.datetime(2019, 12, 31, 12)

    datelist = []
    while start <= end:
        datelist.append(start)
        start += datetime.timedelta(hours=12)

    # 选择下载站点（以上海宝山站 ['58362'] 为例）
    stationfile = pd.read_excel('UPAR_GLB_MUL_FTM_STATION.xlsx')
    stationlist = list(stationfile['区站号'])

    # 设置下载区间 (数据集共计817个站点数据 其中中国站点为 [164: 252])
    # 设置区间起始点 单点下载时间较长 每次下载20个站点
    station_range_start = 172
    station_range_step = 1
    print('下载范围: [{0} - {1})'.format(station_range_start, station_range_start + station_range_step))

    # 批量下载
    for station in stationlist[station_range_start: station_range_start + station_range_step]:
        station = str(station)
        print('正在下载station:' + station)
        for date in tqdm(datelist):
            time = date.strftime('%Y%m%d%H')
            try:
                # 新建文件夹
                dirs = 'data/' + station
                if not os.path.exists(dirs):
                    os.makedirs(dirs)

                # 如文件已存在，则已下载
                filepath = dirs + '/' + station + '_' + time + '.csv'
                if os.path.exists(filepath):
                    # print(date.strftime('%Y%m%d_%H') + '已下载')
                    record_download(station, time, message=True)
                    continue
                else:
                    df = WyomingUpperAir.request_data(date, station)
                    df.to_csv(filepath, index=False)
                record_download(station, time, message=True)
                # print(date.strftime('%Y%m%d_%H') + '下载成功')
            except:
                record_download(station, time)
                # print(date.strftime('%Y%m%d_%H') + '下载失败')
                pass


def test():
    # time = datetime.datetime(2010, 1, 1, 0)
    # station = '58362'
    # while True:
    #     try:
    #         WyomingUpperAir.request_data(time, station)
    #         print(datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '下载正常')
    #     except:
    #         print(datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + 'IP被限制，请尽快处理')
    #         break
    #     sleep(1800)
    time = [datetime.datetime(2010, 1, i, 0) for i in range(1, 6)]
    station = '58362'
    while True:
        _fail = 0
        for t in time:
            try:
                WyomingUpperAir.request_data(t, station)
                print(datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '下载正常')
                break
            except:
                _fail += 1
        if _fail == 3:
            print(datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + 'IP被限制，请尽快处理')
            break
        else:
            sleep(1800)


def refresh():
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
    station_range_start = 230
    station_range_step = 4
    print('下载范围: [{0} - {1})'.format(station_range_start, station_range_start + station_range_step))

    # 批量下载
    for station in stationlist[station_range_start: station_range_start + station_range_step]:
        station = str(station)
        print('正在下载station:' + station)
        for date in tqdm(datelist):
            time = date.strftime('%Y%m%d%H')
            try:
                # 新建文件夹
                dirs = 'data/' + station
                if not os.path.exists(dirs):
                    os.makedirs(dirs)

                #
                download_path = 'data/' + station + '/download.json'
                with open(download_path, 'r+') as f:
                    _download = json.load(f)

                # 如文件已存在，则已下载
                filepath = dirs + '/' + station + '_' + time + '.csv'
                if os.path.exists(filepath):
                    continue
                else:
                    df = WyomingUpperAir.request_data(date, station)
                    df.to_csv(filepath, index=False)
                record_download(station, time, message=True)
                # print(date.strftime('%Y%m%d_%H') + '下载成功')
            except:
                record_download(station, time)
                # print(date.strftime('%Y%m%d_%H') + '下载失败')
                pass


if __name__ == '__main__':
    # 下载
    # download()
    # 测试
    # test()
    # 更新
    refresh()
