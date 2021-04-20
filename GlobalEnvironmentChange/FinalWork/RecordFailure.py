import pandas as pd
import datetime
import json
import os


def record_download(station, time, message=False):
    filepath = 'data/' + station + '/download.json'
    if os.path.exists(filepath):
        _download = json.load(filepath)
    else:
        _download = dict()
    if message:
        _download[time] = -1
    else:
        try:
            _download[time] += 1
        except KeyError:
            _download[time] = 1
    json.dump(_download, filepath)


if __name__ == '__main__':
    import numpy

    start = datetime.datetime(2008, 1, 1, 0)
    end = datetime.datetime(2020, 12, 31, 12)
    datelist = []
    while start <= end:
        datelist.append(start.strftime('%Y%m%d%H'))
        start += datetime.timedelta(hours=12)

    stationfile = pd.read_excel('UPAR_GLB_MUL_FTM_STATION.xlsx')
    stationlist = list(stationfile['区站号'])

    download = pd.DataFrame(numpy.zeros((len(stationlist), len(datelist)), int), columns=datelist, index=stationlist)
    download.to_csv('cache/download.csv')
