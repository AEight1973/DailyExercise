import json
import pandas as pd
import os
import matplotlib.pyplot as plt

data = pd.read_excel('UPAR_GLB_MUL_FTM_STATION.xlsx')
stationlist = data.loc[164: 252, '区站号']


def success_percent():
    condition = []
    for station in list(stationlist):
        filepath = 'data/' + str(station)
        if os.path.exists(filepath):
            with open(filepath + '/download.json', 'r+') as f:
                _download = json.load(f)
            download_all = len(_download)
            download_success = len(os.listdir(filepath)) - 1
            condition.append([station, download_all, download_success, download_success / download_all])
        else:
            condition.append([station, 0, 0, 0])
    pd.DataFrame(condition, columns=['站号', '下载总数', '下载成功', '成功率']).to_csv('cache/condition_percent.csv', encoding='gbk')


def success_total():
    condition = []
    for station in list(stationlist):
        filepath = 'data/' + str(station)
        if os.path.exists(filepath):
            with open(filepath + '/download.json', 'r+') as f:
                _download = json.load(f)
            condition.append(list(_download.values()))
        else:
            condition.append([station, 0, 0, 0])
    pd.DataFrame(condition, columns=list(_download.keys())).to_csv('cache/condition_total.csv', encoding='gbk')


if __name__ == '__main__':
    # 显示各站点下载总数, 下载成功, 成功率
    # success_percent()
    # 显示各站点每天下载情况
    success_total()
    # a = pd.read_csv('cache/condition_total.csv', encoding='gbk')
    # plt.imshow(a)
    # plt.show()
