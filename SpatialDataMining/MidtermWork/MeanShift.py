# -*- coding: utf-8 -*-
import cv2
import gdal
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary, unary_from_softmax, unary_from_labels
import pydensecrf.densecrf as dcrf

# Get im{read,write} from somewhere.
try:
    from cv2 import imread, imwrite
except ImportError:
    # Note that, sadly, skimage unconditionally import scipy and matplotlib,
    # so you'll need them if you don't have OpenCV. But you probably have them.
    from skimage.io import imread, imsave

    imwrite = imsave
    # TODO: Use scipy instead.

from utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian


def read_img(filename):
    dataset = gdal.Open(filename)

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize

    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)

    del dataset
    return im_proj, im_geotrans, im_width, im_height, im_data


def write_img(filename, im_proj, im_geotrans, im_data):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])


if __name__ == '__main__':
    img_path = 'E:/xx/sb_test1.tif'
    im_proj, im_geotrans, im_width, im_height, im_data = read_img(img_path)
    im_data = im_data[0:3, ...]
    im_data = im_data.transpose((2, 1, 0))
    im_temp = im_data.reshape((-1, 3))
    im_temp = np.float32(im_temp)
    bandwidth = estimate_bandwidth(im_temp, quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
    ms.fit_predict(im_temp)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    seg = labels.reshape((im_data.shape[0], im_data.shape[1]))
    seg = seg.transpose(1, 0)
    seg_path = 'E:/xx/test/sb_test1_seg.tif'
    write_img(seg_path, im_proj, im_geotrans, seg)
