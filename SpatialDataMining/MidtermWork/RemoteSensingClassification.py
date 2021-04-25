from GDALTools import *
import numpy as np
import sklearn
import os


class RSImage:
    def __init__(self, _path):
        self.path = _path


def flez():
    return


if __name__ == '__main__':
    img_file = [i for i in os.listdir('data/LC08_L2SP_126058_20200824_20200905_02_T1') if i[-3:] == 'TIF']
    print(img_file)
