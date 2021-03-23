import pandas as pd


def read():
    with open('../data/work01/co2_mm_mlo.txt', 'r') as f:
        data = f.read().split('\n')
    f1 = pd.DataFrame(columns=['year', 'month', 'decimal date', 'monthly average', 'de-season alized', 'days', 'st.dev of days', 'unc. of mon mean'])
    index = 0
    for line in data[53:]:
        f1.loc[index] = line.split()
        index += 1

    with open('../data/work01/data.giss.nasa.gov_gistemp.txt', 'r') as f:
        data = f.read().split('\n')
    f2 = pd.DataFrame(columns=['Year', 'No_Smoothing', 'Lowess(5)'])
    index = 0
    for line in data[6:]:
        f2.loc[index] = line.split()
        index += 1

    return [f1, f2]


if __name__ == '__main__':
    print(read())
