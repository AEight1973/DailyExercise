from osgeo import gdal
from sklearn.preprocessing import MinMaxScaler


# 读图像文件
def read_img(filename):
    dataset = gdal.Open(filename)  # 打开文件

    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    # im_bands = dataset.RasterCount  # 波段数
    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率
    im_proj = dataset.GetProjection()  # 地图投影信息，字符串表示
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)

    # 简化流程，截取其中500 * 500 区域
    im_width = im_height = 500
    im_data = im_data[5900: 6400, 700: 1200]

    del dataset  # 关闭对象dataset，释放内存

    return im_proj, im_geotrans, im_width, im_height, im_data


# 遥感影像的存储
# 写GeoTiff文件
def write_img(filename, im_proj, im_geotrans, im_data):
    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        # 注意数据的存储波段顺序：im_bands, im_height, im_width
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape  # 没看懂

    # 创建文件时 driver = gdal.GetDriverByName("GTiff")，数据类型必须要指定，因为要计算需要多大内存空间。
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset


# TODO 效率待优化
def scaled(img):
    img_1d = list(img.reshape(-1))
    img_1d.sort()
    _zero = img_1d.count(0)
    _min = img_1d[int(0.01 * len(img_1d)) + _zero]
    _max = img_1d[int(0.99 * len(img_1d))]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] <= _min:
                img[i, j] = 0
            elif img[i, j] >= _max:
                img[i, j] = 255
            else:
                img[i, j] = 255 * (img[i, j] - _min) / (_max - _min)
    return img
