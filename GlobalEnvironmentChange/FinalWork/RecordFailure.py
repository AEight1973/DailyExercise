import pandas as pd
import datetime
import json
import os


def record_download(station, time, message='fail'):
    filepath = 'cache/data/' + station + '/download.json'
    if os.path.exists(filepath):
        with open(filepath, 'r+') as f:
            _download = json.load(f)
    else:
        _download = dict()
    if message == 'success':
        _download[time] = 0
    elif message == 'inter':
        _download[time] = -1
    elif message == 'missing':
        _download[time] = -2
    else:
        try:
            if _download[time] < 1:
                _download[time] = 1
            else:
                _download[time] += 1
        except KeyError:
            _download[time] = 1
    with open(filepath, 'w+') as f:
        json.dump(_download, f)


def record_journal(station):
    # 追加记录
    with open('cache/data/' + station + '/download.json', 'r+') as f:
        _download = json.load(f)
    per = len([i for i in list(_download.values()) if i == 0]) / 9484
    with open('cache/record.txt', 'a+') as f:
        text = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '\t' + station + '\t' + str(per) + '\n'
        f.write(text)

# if __name__ == '__main__':
#     # 初始化record文件
#     with open('cache/record.txt', 'w+') as f:
