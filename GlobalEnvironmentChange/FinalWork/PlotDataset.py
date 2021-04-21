import matplotlib.pyplot as plt
import os
import pandas as pd


def dataset_integrity():
    dirs = os.listdir('data')
    table = pd.read_excel('UPAR_GLB_MUL_FTM_STATION.xlsx')

    label, lon, lat, integrity = [], [], [], []
    for s in dirs:
        station = table[table['区站号'] == int(s)]
        download = 'data/' + s + '/download.json'


if __name__ == '__main__':
    dataset_integrity()
