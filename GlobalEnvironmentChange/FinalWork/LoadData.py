import numpy as np
import pandas as pd
import os
from datetime import datetime


def csv2datasets(height=4):
    file_list = os.listdir('data/')
    # features = ['pressure', 'temperature', 'dewpoint', 'direction', 'speed', 'u_wind', 'v_wind']
    features = ['temperature']
    output = pd.DataFrame(columns=features)
    var = [None] * len(features)

    for filename in file_list:
        data = pd.read_csv('data/' + filename)
        select = data[data['height'] == height]
        time = filename.split('.')[0].split('_')[1]
        time = datetime(int(time[0:4]), int(time[4:6]), int(time[6:8]), int(time[8:10]))
        # if time.hour == 12:
        #     continue
        if len(select) == 0 or True in np.isnan(select[features].values[0]):
            output.loc[time] = var
        else:
            var = select[features].values[0]
            output.loc[time] = var
    return output


def pca(data, percent=0.95):
    from sklearn.decomposition import PCA
    _pca = PCA(n_components=percent)
    new_data = _pca.fit_transform(data)
    return new_data


def smooth(x, window_len=100, window='hanning'):
    window = [1/window_len] * window_len
    print(x.T)
    y = np.convolve(x.T[0], window, mode='valid')
    return np.array([y]).T


if __name__ == '__main__':
    a = csv2datasets()
    print(a)
