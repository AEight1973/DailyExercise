import pandas as pd


def record_download(station, time, message=False):
    download = pd.read_csv('cache/download.csv')
    if message:
        download.loc[station, time] = -1
    else:
        download.loc[station, time] += 1


if __name__ == '__main__':
    import numpy
    import datetime

    start = datetime.datetime(2008, 1, 1, 0)
    end = datetime.datetime(2020, 12, 31, 12)
    datelist = []
    while start <= end:
        datelist.append(start)
        start += datetime.timedelta(hours=12)

    stationfile = pd.read_excel('UPAR_GLB_MUL_FTM_STATION.xlsx')
    stationlist = list(stationfile['区站号'])

    download = pd.DataFrame(numpy.zeros((len(stationlist), len(datelist)), int), columns=datelist, index=stationlist)
    download.to_csv('cache/download.csv')
