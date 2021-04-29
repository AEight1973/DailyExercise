from sklearn.cluster import estimate_bandwidth, MeanShift
from GDALTools import *
import numpy as np
import os
import matplotlib.pyplot as plt


class RSImage:
    def __init__(self, _path):
        self.path = _path
        self.band = self._read()
        self.width, self.height = list(self.band.values())[0].shape

    def _read(self):
        _file = [i for i in os.listdir(self.path) if i[-3:] == 'TIF']
        _output = dict()
        for i in _file:
            info = i[: -4].split('_')[-1]
            im_proj, im_geotrans, im_width, im_height, _output[info] = read_img(self.path + '/' + i)
            print('{}读取完成'.format(i))
        # TODO 优化数据的读入
        self.proj = im_proj
        self.geotrans = im_geotrans
        self.shape = (im_width, im_height)
        return _output

    def band_composite(self, _red='B4', _green='B3', _blue='B2'):
        data1 = scaled(self.band[_red])
        data2 = scaled(self.band[_green])
        data3 = scaled(self.band[_blue])
        _img = np.array((data1, data2, data3))
        return _img.transpose((1, 2, 0))


def meanshift(im_data):
    im_temp = im_data.reshape((-1, 3))
    im_temp = np.float32(im_temp)
    print('开始遥感图像分割')
    bandwidth = estimate_bandwidth(im_temp, quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
    ms.fit_predict(im_temp)
    labels = ms.labels_
    # cluster_centers = ms.cluster_centers_
    seg = labels.reshape((im_data.shape[0], im_data.shape[1]))
    return seg


if __name__ == '__main__':
    '''
    数据集
    数据源：Landsat 8 OLI/TIRS 遥感数据
    数据名称：LC08_L2SP_126058_20200824_20200905_02_T1
    数据介绍：马六甲皇京港 (行 126 列 058)
    数据预处理：波段合成(321)
    '''
    # 读取文件
    img = RSImage('data/LC08_L2SP_126058_20200824_20200905_02_T1')

    # 波段合成
    img_real = img.band_composite(_red='B4', _green='B3', _blue='B2')

    plt.imshow(img_real)
    plt.axis('off')
    plt.show()
    print('数据已全部读取完成')

    # 波段合成
    img_com = img.band_composite(_red='B5', _green='B4', _blue='B3')

    plt.imshow(img_com)
    plt.axis('off')
    plt.show()
    print('数据已全部读取完成')

    '''
    数据处理
    方法：遥感图像分割 MeanShift
    '''
    img_trained = meanshift(img_com)
    print('遥感图像分割结束')

    # 遥感图像分割的展示
    plt.imshow(img_trained)
    plt.axis('off')
    plt.show()

    # 遥感图像分割的保存
    seg_path1 = 'result/432.TIFF'
    seg_path2 = 'result/543.TIFF'
    seg_path3 = 'result/result.TIFF'
    write_img(seg_path1, img.proj, img.geotrans, img_real)
    write_img(seg_path2, img.proj, img.geotrans, img_com)
    write_img(seg_path3, img.proj, img.geotrans, img_trained)

    # TODO 不同参数下遥感图像分割的效果展示

    # TODO 继续完善
