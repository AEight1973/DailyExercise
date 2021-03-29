import pandas as pd


def read():
    with open('data/ex1_leaf.txt') as f:
        origin_data = f.read().split('\n')
        data = [[int(i.split()[0]), float(i.split()[1])] for i in origin_data]
        leaf = pd.DataFrame(data, columns=['wave_length', 'reflect_rate'])

    with open('data/AVHRR.txt') as f:
        origin_data = f.read().split('\n')[4:]
        print(origin_data[0].split())
        data = [[int(float(i.split()[0]) * 1000), float(i.split()[1]), float(i.split()[2])] for i in origin_data]
        avhrr = pd.DataFrame(data, columns=['wave_length', 'reflect_rate_1', 'reflect_rate_2'])

    with open('data/MODIS.txt') as f:
        origin_data = f.read().split('\n')[4:]
        data = [[int(float(i.split()[0]) * 1000), float(i.split()[1]), float(i.split()[2])] for i in origin_data]
        modis = pd.DataFrame(data, columns=['wave_length', 'reflect_rate', 'reflect_rate_2'])

    return leaf, avhrr, modis


if __name__ == '__main__':
    print(read())
