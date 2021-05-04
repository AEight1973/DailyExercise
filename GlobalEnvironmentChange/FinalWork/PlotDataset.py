import matplotlib.pyplot as plt
import os
import pandas as pd
import json
# import geopandas as gpd


def dataset_integrity():
    dirs = os.listdir('cache/data')
    table = pd.read_excel('UPAR_GLB_MUL_FTM_STATION.xlsx')

    label, lon, lat, integrity = [], [], [], []
    for s in dirs:
        station = table[table['区站号'] == int(s)]
        path = 'data/' + s + '/download.json'
        if os.path.exists(path):
            with open(path, 'r+') as f:
                download = json.load(f)
        else:
            continue

        label.append(station['台站名'])
        lon.append(station['经度'])
        lat.append(station['纬度'])
        integrity.append(len([v for k, v in download.items() if v == 0]) / len(download))

    # basemap = gpd.read_file('basemap/ne_10m_admin_0_countries.shp')
    # basemap.plot()
    plt.scatter(lon, lat, c=integrity)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    dataset_integrity()
