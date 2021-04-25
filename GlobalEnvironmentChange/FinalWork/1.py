import json
import pandas as pd
import os

data = pd.read_excel('UPAR_GLB_MUL_FTM_STATION.xlsx')
stationlist = data.loc[164: 252, '区站号']

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
condition = pd.DataFrame(condition, columns=['站号', '下载总数', '下载成功', '成功率'])