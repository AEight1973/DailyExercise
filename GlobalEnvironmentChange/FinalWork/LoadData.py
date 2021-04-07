import pandas as pd
import os
from datetime import datetime


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def csv2datasets(height=4):
    file_list = os.listdir('data/')
    features = ['pressure', 'temperature', 'dewpoint', 'direction', 'speed', 'u_wind', 'v_wind']
    output = pd.DataFrame(columns=features)
    var = [None] * len(features)

    for filename in file_list:
        data = pd.read_csv('data/' + filename)
        select = data[data['height'] == height]
        time = filename.split('.')[0].split('_')[1]
        time = datetime(int(time[0:4]), int(time[4:6]), int(time[6:8]), int(time[8:10]))
        if len(select) == 0:
            output.loc[time] = var
        else:
            var = select[features].values[0]
            output.loc[time] = var
    return output


if __name__ == '__main__':
    print(csv2datasets())
