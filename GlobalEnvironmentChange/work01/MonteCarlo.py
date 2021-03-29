import pandas as pd
import numpy as np


def a():
    P_all = []
    for i in range(40):
        P = []
        for j in range(5000):
            xy = np.random.standard_normal(size=(60, 2))
            p = pd.DataFrame(xy).corr()
            P.append(p.iloc[0, 1])
        P.sort()
        P_all.append([P[4499], P[4749], P[4949]])
    P_all = np.array(P_all)
    return np.mean(P_all, axis=0)


if __name__ == '__main__':
    print(a())
