import json
import os
import datetime

import pandas as pd
from tqdm import tqdm

# 2021-5-7

# 将未进行统计的missing数据计入download文件
def debug1():
    from RecordFailure import record_download
    station_list = os.listdir('E:/DataBackup/sounding_station')
    for s in tqdm(station_list):
        path = 'E:/DataBackup/sounding_station/' + s + '/missing'
        if os.path.exists(path):
            record_list = os.listdir(path)
            for r in record_list:
                record_download(s, r.split('_')[1][:-4], 'missing')


# 删除并重新编辑乱码的download文件
def debug2(_station):
    path1 = 'cache/data/' + str(_station)
    path2 = 'E:/DataBackup/sounding_station/' + str(_station)
    if os.path.exists(path1 + '/download.json'):
        os.remove(path1 + '/download.json')
    success = [i.split('_')[1][:-4] for i in os.listdir(path1)] + [i.split('_')[1][:-4] for i in os.listdir(path2)]

    start = datetime.datetime(2008, 1, 1, 0)
    end = datetime.datetime(2019, 12, 31, 12)

    time = []
    while start <= end:
        time.append(start)
        start += datetime.timedelta(hours=12)

    download = {i.strftime('%Y%m%d%H'): 1 for i in time}

    for i in success:
        download[i] = 0

    with open(path1 + '/download.json', 'w+') as f:
        json.dump(download, f)


# 恢复被错误统计的missing数据
def debug3():
    from RecordFailure import record_download
    from sounding_db import csv2db
    station_list = os.listdir('E:/DataBackup/sounding_station')
    for station in station_list:
        data = pd.read_csv('E:/DataBackup/sounding_station/' + station + '/' +
                           os.listdir('E:/DataBackup/sounding_station/' + station)[0])
        pressures = [850, 700, 500, 300, 200]
        tag = None
        for p in pressures:
            if p in list(data['pressure']):
                tag = p
                break
        if os.path.exists('E:/DataBackup/sounding_station/' + station + '/missing'):
            record_list = os.listdir('E:/DataBackup/sounding_station/' + station + '/missing')
            for record in record_list:
                record_download(station, record.split('_')[1][:-4], 'missing')
                csv2db('E:/DataBackup/sounding_station/' + station + '/missing/' + record, tag, record)


if __name__ == '__main__':
    # print('开始执行debug1')
    # debug1()
    # print('开始执行debug2')
    # debug2(57972)
    print('开始执行debug3')
    debug3()
